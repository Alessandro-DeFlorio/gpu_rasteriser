// Per-element procedures of the rasteriser (ADR 0006, 0008, 0009, 0010). Every op of both backends
// is a loop over these functions, so the CPU and CUDA backends agree bit for bit. Each procedure is
// written once for every dimension D (segments, triangles, tetrahedra; R3 §1).
//
// Notation (R1 §2.2): S = 2^k lattice units per voxel; q = round(v·S) are the snapped vertices;
// O = box_lo is the integer voxel origin of the simplex; E_j is the plane function of face j (the
// face opposite vertex j), in lattice units, with E_j(q_j) = D > 0 for a positively oriented
// simplex. A voxel centre X is owned iff t_j = n_j·(X − O) + c_j >= 0 for all j, and π_j = E_j / D.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "tetvox/types.hpp"

namespace tetvox {

__extension__ typedef __int128 int128;

// One simplex's setup record: the element form of SimplexSetupView (fields documented there).
template <Real T, std::size_t D>
  requires SupportedDim<D>
struct SimplexRecord {
  int64_t normal[D + 1][D];
  int64_t offset[D + 1];
  T fraction[D + 1];
  T bary_scale;
  int64_t box_lo[D];
  int64_t box_hi[D];
  SimplexStatus status;
};

// t_j at one voxel centre.
template <std::size_t D>
struct PlaneValues {
  int64_t t[D + 1];
};

// π_0..π_D at one voxel centre.
template <Real T, std::size_t D>
struct Barycentrics {
  T pi[D + 1];
};

// Inclusive interval [first, last] of axis-(D−1) indices; empty when first > last.
struct RowSpan {
  int64_t first;
  int64_t last;
};

namespace detail {

// floor(x / 2^k) and ceil(x / 2^k); >> on a negative value is an arithmetic shift since C++20.
TETVOX_HD inline int64_t floor_shift(int64_t x, int k) { return x >> k; }
TETVOX_HD inline int64_t ceil_shift(int64_t x, int k) { return -((-x) >> k); }

// floor(a / b) and ceil(a / b) for b > 0.
TETVOX_HD inline int64_t floor_div(int64_t a, int64_t b) {
  const int64_t q = a / b;
  return (a % b != 0 && a < 0) ? q - 1 : q;
}
TETVOX_HD inline int64_t ceil_div(int64_t a, int64_t b) { return -floor_div(-a, b); }

TETVOX_HD inline int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }
TETVOX_HD inline int64_t max64(int64_t a, int64_t b) { return a > b ? a : b; }

// x > 0 as a double, rounded to nearest with ties to even (ADR 0008). The compiler's __int128
// conversion is a runtime routine that need not match between host and device, so the value is
// shifted to 63 bits, with every dropped bit folded into a sticky bit 0 (below the rounding
// position: a double keeps 53 bits), converted from int64 and scaled back exactly.
TETVOX_HD inline double to_double_rn(int128 x) {
  const int128 int64_limit = int128{1} << 63;
  if (x < int64_limit) {
    return static_cast<double>(static_cast<int64_t>(x));
  }
  int shift = 0;
  while ((x >> shift) >= int64_limit) {
    ++shift;
  }
  const bool dropped = (x & ((int128{1} << shift) - 1)) != 0;
  const int64_t kept = static_cast<int64_t>(x >> shift) | static_cast<int64_t>(dropped);
  return static_cast<double>(kept) * static_cast<double>(int64_t{1} << shift);
}

// Whether the first non-zero component of n, in axis order 0 … D−1, is positive: face j then owns
// the samples on its plane (ADR 0006 tie-break, axis 0 first per ADR 0007).
template <std::size_t D>
TETVOX_HD bool lex_positive(const int64_t (&n)[D]) {
  for (std::size_t d = 0; d + 1 < D; ++d) {
    if (n[d] != 0) {
      return n[d] > 0;
    }
  }
  return n[D - 1] > 0;
}

// Determinant of an N×N integer matrix, N >= 1, by cofactor expansion along row 0. Exact for the
// minors of setup_simplex (R3 §2.1 bounds them within int64).
template <std::size_t N>
TETVOX_FORCEINLINE TETVOX_HD int64_t det(const int64_t (&m)[N][N]) {
  if constexpr (N == 1) {
    return m[0][0];
  } else {
    int64_t sum = 0;
    for (std::size_t c = 0; c < N; ++c) {
      int64_t sub[N - 1][N - 1];
      for (std::size_t r = 1; r < N; ++r) {
        std::size_t col = 0;
        for (std::size_t c2 = 0; c2 < N; ++c2) {
          if (c2 != c) {
            sub[r - 1][col++] = m[r][c2];
          }
        }
      }
      const int64_t term = m[0][c] * det<N - 1>(sub);
      sum += c % 2 == 0 ? term : -term;
    }
    return sum;
  }
}

// The inward normal of face J (R3 §1.1): (n_J)_m = (−1)^(J+m+1) · the minor of the face edges with
// column m deleted. The edges run from a, the lowest-numbered vertex other than J, to the other
// face vertices in increasing order. In 1D the minor is the empty determinant, 1. J is a template
// parameter and the function is inlined, so that the choice of face vertices folds at compile time
// (GCC otherwise calls it out of line: setup_simplices about 20 % slower than the 3D-only code).
template <std::size_t D, std::size_t J>
TETVOX_FORCEINLINE TETVOX_HD void face_normal(const int64_t (&q)[D + 1][D], int64_t (&n)[D]) {
  if constexpr (D == 1) {
    n[0] = J == 0 ? -1 : 1;
  } else {
    constexpr std::size_t a = J == 0 ? 1 : 0;
    int64_t edges[D - 1][D];
    std::size_t row = 0;
    for (std::size_t v = 0; v <= D; ++v) {
      if (v == J || v == a) {
        continue;
      }
      for (std::size_t d = 0; d < D; ++d) {
        edges[row][d] = q[v][d] - q[a][d];
      }
      ++row;
    }
    for (std::size_t m = 0; m < D; ++m) {
      int64_t minor[D - 1][D - 1];
      for (std::size_t r = 0; r + 1 < D; ++r) {
        std::size_t col = 0;
        for (std::size_t d = 0; d < D; ++d) {
          if (d != m) {
            minor[r][col++] = edges[r][d];
          }
        }
      }
      const int64_t value = det<D - 1>(minor);
      n[m] = (J + m) % 2 == 0 ? -value : value;
    }
  }
}

// The inward normals of all D + 1 faces.
template <std::size_t D, std::size_t... J>
TETVOX_FORCEINLINE TETVOX_HD void face_normals(const int64_t (&q)[D + 1][D], int64_t (&n)[D + 1][D],
                                               std::index_sequence<J...>) {
  (face_normal<D, J>(q, n[J]), ...);
}

}  // namespace detail

// Whether v is finite and |v| <= 2^(62 − k) voxels, the domain of snap (ADR 0008).
TETVOX_HD inline bool in_snap_range(double v, int snap_bits) {
  const double bound = static_cast<double>(int64_t{1} << (snap_range_bits - snap_bits));
  return v >= -bound && v <= bound;  // false for NaN
}

// q = round(v · 2^k), to nearest with ties to even (ADR 0007). The product is exact (power of two),
// so the rounding is the only rounding step. Precondition: in_snap_range(v, snap_bits).
TETVOX_HD inline int64_t snap(double v, int snap_bits) {
  const double scaled = v * static_cast<double>(int64_t{1} << snap_bits);
#if defined(__CUDA_ARCH__)
  return __double2ll_rn(scaled);
#else
  return std::llrint(scaled);  // the default rounding mode: to nearest, ties to even
#endif
}

// Classifies one simplex from its snapped vertices q[0..D] and computes its record (ADR 0008
// order, after the snap-range check that the caller does on the unsnapped positions):
//   1. outside_window if, on some axis, max q − min q > 2^(k+L), L = window_bits<D>(k);
//   2. inverted if D <= 0;
//   3. ok: n_j, c_j = floor((E_j(S·O) − 1 + own_j) / S), f_j = (E_j(S·O) − S·c_j) / S ∈ [0, 1],
//      2^k / D (R1 §2.2, §2.5).
// For a simplex that is not ok, only status is set.
template <Real T, std::size_t D>
TETVOX_HD SimplexRecord<T, D> setup_simplex(const int64_t (&q)[D + 1][D], int snap_bits) {
  SimplexRecord<T, D> rec{};
  const int k = snap_bits;
  const uint64_t window = uint64_t{1} << (k + window_bits<D>(k));
  for (std::size_t d = 0; d < D; ++d) {
    int64_t lo = q[0][d];
    int64_t hi = q[0][d];
    for (std::size_t v = 1; v <= D; ++v) {
      lo = detail::min64(lo, q[v][d]);
      hi = detail::max64(hi, q[v][d]);
    }
    // Unsigned: with |q| <= 2^62 the span may reach 2^63, one more than int64 holds.
    if (static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo) > window) {
      rec.status = SimplexStatus::outside_window;
      return rec;
    }
    rec.box_lo[d] = detail::ceil_shift(lo, k);
    rec.box_hi[d] = detail::floor_shift(hi, k);
  }

  detail::face_normals<D>(q, rec.normal, std::make_index_sequence<D + 1>{});
  // E_D(q_D) = n_D · (q_D − q_0).
  int128 det = 0;
  for (std::size_t d = 0; d < D; ++d) {
    det += int128{rec.normal[D][d]} * (q[D][d] - q[0][d]);
  }
  if (det <= 0) {
    rec.status = SimplexStatus::inverted;
    return rec;
  }

  const int64_t scale = int64_t{1} << k;
  const T inv_scale = static_cast<T>(1.0 / static_cast<double>(scale));  // exact: 2^-k, k <= 24
  for (std::size_t j = 0; j <= D; ++j) {
    const int64_t (&a)[D] = q[j == 0 ? 1 : 0];  // a vertex on face j
    int128 plane_at_origin = 0;                 // E_j(S·O)
    for (std::size_t d = 0; d < D; ++d) {
      plane_at_origin += int128{rec.normal[j][d]} * (scale * rec.box_lo[d] - a[d]);
    }
    const int own = detail::lex_positive<D>(rec.normal[j]) ? 1 : 0;
    const int128 offset = (plane_at_origin - 1 + own) >> k;
    const int64_t remainder = static_cast<int64_t>(plane_at_origin - offset * scale);  // [0, S]
    rec.offset[j] = static_cast<int64_t>(offset);
    rec.fraction[j] = static_cast<T>(remainder) * inv_scale;
  }
  rec.bary_scale = static_cast<T>(static_cast<double>(scale) / detail::to_double_rn(det));
  rec.status = SimplexStatus::ok;
  return rec;
}

// t_j = n_j · (X − O) + c_j at voxel centre X, exact in int64 for X in the simplex's box
// (ADR 0006).
template <Real T, std::size_t D>
TETVOX_HD PlaneValues<D> plane_values(const SimplexRecord<T, D>& rec, const int64_t (&x)[D]) {
  PlaneValues<D> p{};
  for (std::size_t j = 0; j <= D; ++j) {
    int64_t t = rec.offset[j];
    for (std::size_t d = 0; d < D; ++d) {
      t += rec.normal[j][d] * (x[d] - rec.box_lo[d]);
    }
    p.t[j] = t;
  }
  return p;
}

// The inside test: all t_j >= 0 (the tie-break is folded into c_j), that is, the OR of the t_j has
// its sign bit clear.
template <std::size_t D>
TETVOX_HD bool owns(const PlaneValues<D>& p) {
  int64_t any = 0;
  for (std::size_t j = 0; j <= D; ++j) {
    any |= p.t[j];
  }
  return any >= 0;
}

// π_j = (T(t_j) + f_j) · 2^k / D (R1 §2.5, ADR 0008). Every π_j >= 0 where owns(p).
template <Real T, std::size_t D>
TETVOX_HD Barycentrics<T, D> barycentric(const SimplexRecord<T, D>& rec, const PlaneValues<D>& p) {
  Barycentrics<T, D> b{};
  for (std::size_t j = 0; j <= D; ++j) {
    b.pi[j] = (static_cast<T>(p.t[j]) + rec.fraction[j]) * rec.bary_scale;
  }
  return b;
}

// The indices along the last axis, D − 1, owned by the simplex in the box row through x (x[D−1]
// is ignored), clipped to the box and to [0, extent). Rows run along the contiguous axis
// (ADR 0010). Exact: t_j(i) = n_j,D−1 · (i − O_D−1) + b_j >= 0 is solved per face with ceil_div and
// floor_div. Precondition: x[0 … D−2] lie in the simplex's box.
template <Real T, std::size_t D>
TETVOX_FORCEINLINE TETVOX_HD RowSpan row_span(const SimplexRecord<T, D>& rec, const int64_t (&x)[D],
                                              int64_t extent) {
  constexpr std::size_t axis = D - 1;
  RowSpan s{detail::max64(rec.box_lo[axis], 0), detail::min64(rec.box_hi[axis], extent - 1)};
  for (std::size_t j = 0; j <= D; ++j) {
    const int64_t a = rec.normal[j][axis];
    int64_t b = rec.offset[j];
    for (std::size_t d = 0; d < axis; ++d) {
      b += rec.normal[j][d] * (x[d] - rec.box_lo[d]);
    }
    if (a > 0) {
      s.first = detail::max64(s.first, rec.box_lo[axis] + detail::ceil_div(-b, a));
    } else if (a < 0) {
      s.last = detail::min64(s.last, rec.box_lo[axis] + detail::floor_div(b, -a));
    } else if (b < 0) {
      return RowSpan{0, -1};
    }
  }
  return s;
}

namespace detail {

// Gather and scatter one record from and to the structure-of-arrays view.
template <Real T, std::size_t D>
TETVOX_HD SimplexRecord<T, D> load_record(const SimplexSetupView<T, D>& s, int64_t i) {
  SimplexRecord<T, D> rec{};
  rec.status = s.status[i];
  for (std::size_t j = 0; j <= D; ++j) {
    const int64_t face = i * static_cast<int64_t>(D + 1) + static_cast<int64_t>(j);
    for (std::size_t d = 0; d < D; ++d) {
      rec.normal[j][d] = s.normals[face * static_cast<int64_t>(D) + static_cast<int64_t>(d)];
    }
    rec.offset[j] = s.offsets[face];
    rec.fraction[j] = s.fractions[face];
  }
  rec.bary_scale = s.bary_scale[i];
  for (std::size_t d = 0; d < D; ++d) {
    rec.box_lo[d] = s.box_lo[i * static_cast<int64_t>(D) + static_cast<int64_t>(d)];
    rec.box_hi[d] = s.box_hi[i * static_cast<int64_t>(D) + static_cast<int64_t>(d)];
  }
  return rec;
}

template <Real T, std::size_t D>
TETVOX_HD void store_record(const SimplexSetupView<T, D>& s, int64_t i,
                            const SimplexRecord<T, D>& rec) {
  s.status[i] = rec.status;
  if (rec.status != SimplexStatus::ok) {
    return;
  }
  for (std::size_t j = 0; j <= D; ++j) {
    const int64_t face = i * static_cast<int64_t>(D + 1) + static_cast<int64_t>(j);
    for (std::size_t d = 0; d < D; ++d) {
      s.normals[face * static_cast<int64_t>(D) + static_cast<int64_t>(d)] = rec.normal[j][d];
    }
    s.offsets[face] = rec.offset[j];
    s.fractions[face] = rec.fraction[j];
  }
  s.bary_scale[i] = rec.bary_scale;
  for (std::size_t d = 0; d < D; ++d) {
    s.box_lo[i * static_cast<int64_t>(D) + static_cast<int64_t>(d)] = rec.box_lo[d];
    s.box_hi[i * static_cast<int64_t>(D) + static_cast<int64_t>(d)] = rec.box_hi[d];
  }
}

}  // namespace detail

}  // namespace tetvox
