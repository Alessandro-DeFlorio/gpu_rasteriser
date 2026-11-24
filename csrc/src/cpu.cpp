// CPU backend of the rasterisation ops (ADR 0007, 0009, 0010).
//
// The loop structure follows meshing_cxx, the earlier CPU rasteriser by Koen Van Leemput and
// Alessandro De Florio, used here under MIT with their permission
// (docs/knowledge/meshing-cxx-reference.md): OpenMP over simplices, each scanning its grid-clipped
// bounding box. Departures from it:
//   - snapping rounds to nearest with ties to even at scale 2^k (it truncates toward zero at 10^4);
//   - the tie-break puts axis 0 first (it scans axis 2 first);
//   - the per-voxel test is int64 with the tie-break folded into c_j, after an __int128 setup and a
//     per-simplex window check (it can overflow int64 silently);
//   - π comes from the snapped integers (it recomputes π in double from the unsnapped vertices);
//   - each box row, along the last axis, is solved in closed form by row_span;
//   - inverted, degenerate and out-of-window simplices are classified and reported (it skips D < 0
//     silently and rasterises D = 0);
//   - every op is written once for any dimension D (it is 3D only).
#include "tetvox/cpu.hpp"

#include <atomic>
#include <climits>
#include <stdexcept>

#include "tetvox/raster.hpp"

namespace tetvox::cpu {
namespace {

// Simplices per OpenMP work item. Box sizes vary widely, so simplices are handed out dynamically.
constexpr int simplex_chunk = 64;

template <std::size_t D>
void check_shape(const GridShape<D>& shape) {
  for (const int64_t extent : shape) {
    if (extent < 0) {
      throw std::invalid_argument("grid shape has a negative extent");
    }
  }
}

// The number of rows of the grid: the product of the extents of axes 0 … D−2 (1 in 1D).
template <std::size_t D>
int64_t num_rows(const GridShape<D>& shape) {
  int64_t rows = 1;
  for (std::size_t d = 0; d + 1 < D; ++d) {
    rows *= shape[d];
  }
  return rows;
}

template <std::size_t D>
int64_t num_voxels(const GridShape<D>& shape) {
  return num_rows(shape) * shape[D - 1];
}

// The C-order element index of voxel x.
template <std::size_t D>
inline int64_t voxel_index(const GridShape<D>& shape, const int64_t (&x)[D]) {
  int64_t index = x[0];
  for (std::size_t d = 1; d < D; ++d) {
    index = index * shape[d] + x[d];
  }
  return index;
}

// x[0 … D−2] of grid row r, rows numbered in C order.
template <std::size_t D>
inline void row_indices(const GridShape<D>& shape, int64_t r, int64_t (&x)[D]) {
  for (std::size_t d = D - 1; d-- > 0;) {
    x[d] = r % shape[d];
    r /= shape[d];
  }
}

template <class V>
void fill(V* data, int64_t n, V value) {
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; ++i) {
    data[i] = value;
  }
}

// Concurrent writers only meet where the mesh is not a tiling; the result is then unspecified, but
// relaxed atomics keep it free of data races (ADR 0008).
template <class V>
inline void store_relaxed(V& target, V value) {
  std::atomic_ref<V>(target).store(value, std::memory_order_relaxed);
}

// The simplex's box clipped to the grid, inclusive bounds (empty on an axis when lo > hi).
template <std::size_t D>
struct Box {
  int64_t lo[D];
  int64_t hi[D];
};

template <Real T, std::size_t D>
inline Box<D> clipped_box(const SimplexRecord<T, D>& rec, const GridShape<D>& shape) {
  Box<D> box{};
  for (std::size_t d = 0; d < D; ++d) {
    box.lo[d] = detail::max64(rec.box_lo[d], 0);
    box.hi[d] = detail::min64(rec.box_hi[d], shape[d] - 1);
  }
  return box;
}

// Calls body(i_0, …, i_{Axes−1}) for every index tuple of the box on axes 0 … Axes−1, in C order.
// Each level owns its loop variable and passes the indices on as values, which keeps them in
// registers (R3 §4.3).
template <std::size_t Axes, std::size_t D, class Body, class... Is>
inline void for_each_in_box(const Box<D>& box, const Body& body, Is... is) {
  constexpr std::size_t axis = sizeof...(Is);
  if constexpr (axis == Axes) {
    body(is...);
  } else {
    for (int64_t i = box.lo[axis]; i <= box.hi[axis]; ++i) {
      for_each_in_box<Axes>(box, body, is..., i);
    }
  }
}

// Calls visit(x, index) for every voxel centre x the (ok) simplex owns, one box row at a time.
template <Real T, std::size_t D, class Visit>
inline void for_each_owned(const SimplexRecord<T, D>& rec, const GridShape<D>& shape,
                           const Visit& visit) {
  const Box<D> box = clipped_box(rec, shape);
  for_each_in_box<D - 1>(box, [&](auto... lead) {
    const int64_t row[D] = {lead..., 0};
    const RowSpan span = row_span(rec, row, shape[D - 1]);
    const int64_t row_start = voxel_index(shape, row);
    for (int64_t i = span.first; i <= span.last; ++i) {
      const int64_t x[D] = {lead..., i};
      visit(x, row_start + i);
    }
  });
}

// Provisional check of a caller's simplex_id (ROADMAP open question): −1, or an ok simplex whose
// box holds the voxel, so that plane_values stays inside the int64 budget.
template <Real T, std::size_t D>
bool valid_simplex_id(const SimplexSetupView<T, D>& s, int32_t t, const int64_t (&x)[D]) {
  if (t == -1) {
    return true;
  }
  if (t < 0 || t >= s.num_simplices || s.status[t] != SimplexStatus::ok) {
    return false;
  }
  for (std::size_t d = 0; d < D; ++d) {
    const int64_t i = t * static_cast<int64_t>(D) + static_cast<int64_t>(d);
    if (x[d] < s.box_lo[i] || x[d] > s.box_hi[i]) {
      return false;
    }
  }
  return true;
}

template <Real T, std::size_t D>
inline void store_barycentrics(const SimplexRecord<T, D>& rec, const int64_t (&x)[D], T* bary) {
  const Barycentrics<T, D> b = barycentric(rec, plane_values(rec, x));
  for (std::size_t j = 1; j <= D; ++j) {
    store_relaxed(bary[j - 1], b.pi[j]);
  }
}

}  // namespace

template <Real T, std::size_t D>
void setup_simplices(const MeshView<T, D>& mesh, int snap_bits, const SimplexSetupView<T, D>& out) {
  if (snap_bits < 0 || snap_bits > max_snap_bits) {
    throw std::invalid_argument("snap_bits must lie in [0, 24]");
  }
  if (mesh.num_vertices < 0) {
    throw std::invalid_argument("num_vertices is negative");
  }
  if (!(0 <= mesh.num_simplices && mesh.num_simplices <= INT32_MAX)) {
    throw std::invalid_argument("num_simplices must lie in [0, 2^31 - 1]");
  }
  constexpr int64_t corners = D + 1;
  int64_t bad_indices = 0;
#pragma omp parallel for schedule(static) reduction(+ : bad_indices)
  for (int64_t i = 0; i < mesh.num_simplices * corners; ++i) {
    const int32_t v = mesh.simplices[i];
    bad_indices += (v < 0 || v >= mesh.num_vertices) ? 1 : 0;
  }
  if (bad_indices != 0) {
    throw std::invalid_argument("a simplex index lies outside [0, num_vertices)");
  }

#pragma omp parallel for schedule(static)
  for (int64_t t = 0; t < mesh.num_simplices; ++t) {
    double p[D + 1][D];
    bool in_range = true;
    for (std::size_t v = 0; v <= D; ++v) {
      const int64_t vertex = mesh.simplices[t * corners + static_cast<int64_t>(v)];
      for (std::size_t d = 0; d < D; ++d) {
        p[v][d] = static_cast<double>(
            mesh.positions[vertex * static_cast<int64_t>(D) + static_cast<int64_t>(d)]);
        in_range = in_range && in_snap_range(p[v][d], snap_bits);
      }
    }
    if (!in_range) {
      out.status[t] = SimplexStatus::outside_window;
      continue;
    }
    int64_t q[D + 1][D];
    for (std::size_t v = 0; v <= D; ++v) {
      for (std::size_t d = 0; d < D; ++d) {
        q[v][d] = snap(p[v][d], snap_bits);
      }
    }
    detail::store_record(out, t, setup_simplex<T, D>(q, snap_bits));
  }
}

SkippedSimplices count_skipped(const SimplexStatus* status, int64_t num_simplices) {
  int64_t inverted = 0;
  int64_t outside_window = 0;
#pragma omp parallel for schedule(static) reduction(+ : inverted, outside_window)
  for (int64_t t = 0; t < num_simplices; ++t) {
    inverted += status[t] == SimplexStatus::inverted ? 1 : 0;
    outside_window += status[t] == SimplexStatus::outside_window ? 1 : 0;
  }
  return SkippedSimplices{inverted, outside_window};
}

template <Real T, std::size_t D>
void locate(const SimplexSetupView<T, D>& s, GridShape<D> shape, int32_t* simplex_id) {
  check_shape(shape);
  fill(simplex_id, num_voxels(shape), int32_t{-1});
#pragma omp parallel for schedule(dynamic, simplex_chunk)
  for (int64_t t = 0; t < s.num_simplices; ++t) {
    const SimplexRecord<T, D> rec = detail::load_record(s, t);
    if (rec.status != SimplexStatus::ok) {
      continue;
    }
    for_each_owned(rec, shape, [&](const int64_t (&)[D], int64_t index) {
      store_relaxed(simplex_id[index], static_cast<int32_t>(t));
    });
  }
}

template <Real T, std::size_t D>
void barycentrics(const SimplexSetupView<T, D>& s, GridShape<D> shape, const int32_t* simplex_id,
                  T* bary) {
  check_shape(shape);
  const int64_t rows = num_rows(shape);
  const int64_t width = shape[D - 1];
  int64_t bad_ids = 0;
#pragma omp parallel for schedule(static) reduction(+ : bad_ids)
  for (int64_t r = 0; r < rows; ++r) {
    int64_t x[D];
    row_indices(shape, r, x);
    for (int64_t i = 0; i < width; ++i) {
      x[D - 1] = i;
      bad_ids += valid_simplex_id(s, simplex_id[r * width + i], x) ? 0 : 1;
    }
  }
  if (bad_ids != 0) {
    throw std::invalid_argument(
        "simplex_id holds a value outside [-1, num_simplices), a skipped simplex, or a simplex "
        "whose box does not contain the voxel");
  }

#pragma omp parallel for schedule(static)
  for (int64_t r = 0; r < rows; ++r) {
    int64_t x[D];
    row_indices(shape, r, x);
    for (int64_t i = 0; i < width; ++i) {
      const int64_t index = r * width + i;
      const int32_t t = simplex_id[index];
      if (t < 0) {
        for (std::size_t j = 0; j < D; ++j) {
          bary[index * static_cast<int64_t>(D) + static_cast<int64_t>(j)] = T{0};
        }
        continue;
      }
      x[D - 1] = i;
      store_barycentrics(detail::load_record(s, t), x, bary + index * static_cast<int64_t>(D));
    }
  }
}

template <Real T, std::size_t D>
void rasterize(const SimplexSetupView<T, D>& s, GridShape<D> shape, int32_t* simplex_id, T* bary) {
  check_shape(shape);
  fill(simplex_id, num_voxels(shape), int32_t{-1});
  fill(bary, num_voxels(shape) * static_cast<int64_t>(D), T{0});
#pragma omp parallel for schedule(dynamic, simplex_chunk)
  for (int64_t t = 0; t < s.num_simplices; ++t) {
    const SimplexRecord<T, D> rec = detail::load_record(s, t);
    if (rec.status != SimplexStatus::ok) {
      continue;
    }
    for_each_owned(rec, shape, [&](const int64_t (&x)[D], int64_t index) {
      store_relaxed(simplex_id[index], static_cast<int32_t>(t));
      store_barycentrics(rec, x, bary + index * static_cast<int64_t>(D));
    });
  }
}

template <Real T, std::size_t D>
void count_claims(const SimplexSetupView<T, D>& s, GridShape<D> shape, int32_t* claims) {
  check_shape(shape);
  fill(claims, num_voxels(shape), int32_t{0});
#pragma omp parallel for schedule(dynamic, simplex_chunk)
  for (int64_t t = 0; t < s.num_simplices; ++t) {
    const SimplexRecord<T, D> rec = detail::load_record(s, t);
    if (rec.status != SimplexStatus::ok) {
      continue;
    }
    for_each_in_box<D>(clipped_box(rec, shape), [&](auto... all) {
      const int64_t x[D] = {all...};
      if (owns(plane_values(rec, x))) {
        std::atomic_ref<int32_t>(claims[voxel_index(shape, x)])
            .fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
}

#define TETVOX_INSTANTIATE_CPU(T, D)                                                              \
  template void setup_simplices<T, D>(const MeshView<T, D>&, int, const SimplexSetupView<T, D>&); \
  template void locate<T, D>(const SimplexSetupView<T, D>&, GridShape<D>, int32_t*);              \
  template void barycentrics<T, D>(const SimplexSetupView<T, D>&, GridShape<D>, const int32_t*,   \
                                   T*);                                                           \
  template void rasterize<T, D>(const SimplexSetupView<T, D>&, GridShape<D>, int32_t*, T*);       \
  template void count_claims<T, D>(const SimplexSetupView<T, D>&, GridShape<D>, int32_t*);
TETVOX_INSTANTIATE_CPU(float, 1)
TETVOX_INSTANTIATE_CPU(float, 2)
TETVOX_INSTANTIATE_CPU(float, 3)
TETVOX_INSTANTIATE_CPU(double, 1)
TETVOX_INSTANTIATE_CPU(double, 2)
TETVOX_INSTANTIATE_CPU(double, 3)
#undef TETVOX_INSTANTIATE_CPU

}  // namespace tetvox::cpu
