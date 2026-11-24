// Per-element procedures of the rasteriser (ADR 0006, 0008, 0009). Every op of both backends is a
// loop over these functions, so the CPU and CUDA backends agree bit for bit.
//
// Notation (R1 §2.2): S = 2^k lattice units per voxel; q = round(v·S) are the snapped vertices;
// O = box_lo is the integer voxel origin of the tet; E_j is the plane function of face j (the face
// opposite vertex j), in lattice units, with E_j(q_j) = D > 0 for a positively oriented tet.
// A voxel centre X is owned iff t_j = n_j·(X − O) + c_j >= 0 for all j, and π_j = E_j / D.
#pragma once

#include <cmath>
#include <cstdint>

#include "tetvox/types.hpp"

namespace tetvox {

__extension__ typedef __int128 int128;

// One tet's setup record: the element form of TetSetupView (fields documented there).
template <Real T>
struct TetRecord {
  int64_t normal[4][3];
  int64_t offset[4];
  T fraction[4];
  T bary_scale;
  int64_t box_lo[3];
  int64_t box_hi[3];
  TetStatus status;
};

// t_j at one voxel centre.
struct PlaneValues {
  int64_t t[4];
};

// π_0..π_3 at one voxel centre.
template <Real T>
struct Barycentrics {
  T pi[4];
};

// Inclusive interval [first, last] of axis-0 indices; empty when first > last.
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

// Whether the first non-zero component of n, in axis order 0, 1, 2, is positive: face j then owns
// the samples on its plane (ADR 0006 tie-break, axis 0 first per ADR 0007).
TETVOX_HD inline bool lex_positive(const int64_t (&n)[3]) {
  if (n[0] != 0) {
    return n[0] > 0;
  }
  if (n[1] != 0) {
    return n[1] > 0;
  }
  return n[2] > 0;
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

// Classifies one tet from its snapped vertices and computes its record (ADR 0008 order, after the
// snap-range check that the caller does on the unsnapped positions):
//   1. outside_window if, on some axis, max q − min q > 2^(k+L);
//   2. inverted if D <= 0;
//   3. ok: n_j, c_j = floor((E_j(S·O) − 1 + own_j) / S), f_j = (E_j(S·O) − S·c_j) / S ∈ [0, 1],
//      2^k / D (R1 §2.2, §2.5).
// For a tet that is not ok, only status is set.
template <Real T>
TETVOX_HD TetRecord<T> setup_tet(const int64_t (&q)[4][3], int snap_bits) {
  TetRecord<T> rec{};
  const int k = snap_bits;
  const uint64_t window = uint64_t{1} << (k + window_bits(k));
  for (int d = 0; d < 3; ++d) {
    int64_t lo = q[0][d];
    int64_t hi = q[0][d];
    for (int v = 1; v < 4; ++v) {
      lo = detail::min64(lo, q[v][d]);
      hi = detail::max64(hi, q[v][d]);
    }
    // Unsigned: with |q| <= 2^62 the span may reach 2^63, one more than int64 holds.
    if (static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo) > window) {
      rec.status = TetStatus::outside_window;
      return rec;
    }
    rec.box_lo[d] = detail::ceil_shift(lo, k);
    rec.box_hi[d] = detail::floor_shift(hi, k);
  }

  // (a, b, c) of the face opposite vertex j, ordered so that n_j = (b − a) × (c − a) points into a
  // positively oriented tet (R1, "Recommended algorithm").
  const int faces[4][3] = {{1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}};
  for (int j = 0; j < 4; ++j) {
    const int64_t (&a)[3] = q[faces[j][0]];
    const int64_t (&b)[3] = q[faces[j][1]];
    const int64_t (&c)[3] = q[faces[j][2]];
    const int64_t e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const int64_t e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    rec.normal[j][0] = e1[1] * e2[2] - e1[2] * e2[1];
    rec.normal[j][1] = e1[2] * e2[0] - e1[0] * e2[2];
    rec.normal[j][2] = e1[0] * e2[1] - e1[1] * e2[0];
  }
  int128 det = 0;
  for (int d = 0; d < 3; ++d) {
    det += int128{rec.normal[3][d]} * (q[3][d] - q[0][d]);
  }
  if (det <= 0) {
    rec.status = TetStatus::inverted;
    return rec;
  }

  const int64_t scale = int64_t{1} << k;
  const T inv_scale = static_cast<T>(1.0 / static_cast<double>(scale));  // exact: 2^-k, k <= 24
  for (int j = 0; j < 4; ++j) {
    const int64_t (&a)[3] = q[faces[j][0]];
    int128 plane_at_origin = 0;  // E_j(S·O)
    for (int d = 0; d < 3; ++d) {
      plane_at_origin += int128{rec.normal[j][d]} * (scale * rec.box_lo[d] - a[d]);
    }
    const int own = detail::lex_positive(rec.normal[j]) ? 1 : 0;
    const int128 offset = (plane_at_origin - 1 + own) >> k;
    const int64_t remainder = static_cast<int64_t>(plane_at_origin - offset * scale);  // [0, S]
    rec.offset[j] = static_cast<int64_t>(offset);
    rec.fraction[j] = static_cast<T>(remainder) * inv_scale;
  }
  rec.bary_scale = static_cast<T>(static_cast<double>(scale) / detail::to_double_rn(det));
  rec.status = TetStatus::ok;
  return rec;
}

// t_j = n_j · (X − O) + c_j at voxel centre X, exact in int64 for X in the tet's box (ADR 0006).
template <Real T>
TETVOX_HD PlaneValues plane_values(const TetRecord<T>& rec, const int64_t (&x)[3]) {
  PlaneValues p{};
  for (int j = 0; j < 4; ++j) {
    int64_t t = rec.offset[j];
    for (int d = 0; d < 3; ++d) {
      t += rec.normal[j][d] * (x[d] - rec.box_lo[d]);
    }
    p.t[j] = t;
  }
  return p;
}

// The inside test: all t_j >= 0 (the tie-break is folded into c_j).
TETVOX_HD inline bool owns(const PlaneValues& p) {
  return p.t[0] >= 0 && p.t[1] >= 0 && p.t[2] >= 0 && p.t[3] >= 0;
}

// π_j = (T(t_j) + f_j) · 2^k / D (R1 §2.5, ADR 0008). Every π_j >= 0 where owns(p).
template <Real T>
TETVOX_HD Barycentrics<T> barycentric(const TetRecord<T>& rec, const PlaneValues& p) {
  Barycentrics<T> b{};
  for (int j = 0; j < 4; ++j) {
    b.pi[j] = (static_cast<T>(p.t[j]) + rec.fraction[j]) * rec.bary_scale;
  }
  return b;
}

// The axis-0 indices owned by the tet in the box row (·, i1, i2), clipped to the box and to
// [0, extent0). Exact: t_j(i0) = n_j0 · (i0 − O_0) + b_j >= 0 is solved per face with ceil_div and
// floor_div. Precondition: i1 and i2 lie in the tet's box.
template <Real T>
TETVOX_HD RowSpan row_span(const TetRecord<T>& rec, int64_t i1, int64_t i2, int64_t extent0) {
  RowSpan s{detail::max64(rec.box_lo[0], 0), detail::min64(rec.box_hi[0], extent0 - 1)};
  const int64_t r1 = i1 - rec.box_lo[1];
  const int64_t r2 = i2 - rec.box_lo[2];
  for (int j = 0; j < 4; ++j) {
    const int64_t a = rec.normal[j][0];
    const int64_t b = rec.normal[j][1] * r1 + rec.normal[j][2] * r2 + rec.offset[j];
    if (a > 0) {
      s.first = detail::max64(s.first, rec.box_lo[0] + detail::ceil_div(-b, a));
    } else if (a < 0) {
      s.last = detail::min64(s.last, rec.box_lo[0] + detail::floor_div(b, -a));
    } else if (b < 0) {
      return RowSpan{0, -1};
    }
  }
  return s;
}

namespace detail {

// Gather and scatter one record from and to the structure-of-arrays view.
template <Real T>
TETVOX_HD TetRecord<T> load_record(const TetSetupView<T>& s, int64_t tet) {
  TetRecord<T> rec{};
  rec.status = s.status[tet];
  for (int j = 0; j < 4; ++j) {
    for (int d = 0; d < 3; ++d) {
      rec.normal[j][d] = s.normals[(tet * 4 + j) * 3 + d];
    }
    rec.offset[j] = s.offsets[tet * 4 + j];
    rec.fraction[j] = s.fractions[tet * 4 + j];
  }
  rec.bary_scale = s.bary_scale[tet];
  for (int d = 0; d < 3; ++d) {
    rec.box_lo[d] = s.box_lo[tet * 3 + d];
    rec.box_hi[d] = s.box_hi[tet * 3 + d];
  }
  return rec;
}

template <Real T>
TETVOX_HD void store_record(const TetSetupView<T>& s, int64_t tet, const TetRecord<T>& rec) {
  s.status[tet] = rec.status;
  if (rec.status != TetStatus::ok) {
    return;
  }
  for (int j = 0; j < 4; ++j) {
    for (int d = 0; d < 3; ++d) {
      s.normals[(tet * 4 + j) * 3 + d] = rec.normal[j][d];
    }
    s.offsets[tet * 4 + j] = rec.offset[j];
    s.fractions[tet * 4 + j] = rec.fraction[j];
  }
  s.bary_scale[tet] = rec.bary_scale;
  for (int d = 0; d < 3; ++d) {
    s.box_lo[tet * 3 + d] = rec.box_lo[d];
    s.box_hi[tet * 3 + d] = rec.box_hi[d];
  }
}

}  // namespace detail

}  // namespace tetvox
