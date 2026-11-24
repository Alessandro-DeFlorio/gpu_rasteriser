// CPU backend of the rasterisation ops (ADR 0007, 0009).
//
// The loop structure follows meshing_cxx, the earlier CPU rasteriser by Koen Van Leemput and
// Alessandro De Florio, used here under MIT with their permission
// (docs/knowledge/meshing-cxx-reference.md): OpenMP over tets, each scanning its grid-clipped
// bounding box. Departures from it:
//   - snapping rounds to nearest with ties to even at scale 2^k (it truncates toward zero at 10^4);
//   - the tie-break puts axis 0 first (it scans axis 2 first);
//   - the per-voxel test is int64 with the tie-break folded into c_j, after an __int128 setup and a
//     per-tet window check (it can overflow int64 silently);
//   - π comes from the snapped integers (it recomputes π in double from the unsnapped vertices);
//   - each box row is solved in closed form by row_span;
//   - inverted, degenerate and out-of-window tets are classified and reported (it skips D < 0
//     silently and rasterises D = 0).
#include "tetvox/cpu.hpp"

#include <atomic>
#include <climits>
#include <stdexcept>

#include "tetvox/raster.hpp"

namespace tetvox::cpu {
namespace {

// Tets per OpenMP work item. Box sizes vary widely, so tets are handed out dynamically.
constexpr int tet_chunk = 64;

void check_shape(GridShape shape) {
  for (const int64_t extent : shape) {
    if (extent < 0) {
      throw std::invalid_argument("grid shape has a negative extent");
    }
  }
}

int64_t num_voxels(GridShape shape) { return shape[0] * shape[1] * shape[2]; }

int64_t voxel_index(GridShape shape, int64_t i0, int64_t i1, int64_t i2) {
  return (i0 * shape[1] + i1) * shape[2] + i2;
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
void store_relaxed(V& target, V value) {
  std::atomic_ref<V>(target).store(value, std::memory_order_relaxed);
}

// The tet's box clipped to the grid, inclusive bounds (empty on an axis when lo > hi).
struct Box {
  int64_t lo[3];
  int64_t hi[3];
};

template <Real T>
Box clipped_box(const TetRecord<T>& rec, GridShape shape) {
  Box box{};
  for (int d = 0; d < 3; ++d) {
    box.lo[d] = detail::max64(rec.box_lo[d], 0);
    box.hi[d] = detail::min64(rec.box_hi[d], shape[d] - 1);
  }
  return box;
}

// Calls visit(x, index) for every voxel centre x the (ok) tet owns, one box row at a time.
template <Real T, class Visit>
void for_each_owned(const TetRecord<T>& rec, GridShape shape, Visit&& visit) {
  const Box box = clipped_box(rec, shape);
  for (int64_t i1 = box.lo[1]; i1 <= box.hi[1]; ++i1) {
    for (int64_t i2 = box.lo[2]; i2 <= box.hi[2]; ++i2) {
      const RowSpan span = row_span(rec, i1, i2, shape[0]);
      for (int64_t i0 = span.first; i0 <= span.last; ++i0) {
        const int64_t x[3] = {i0, i1, i2};
        visit(x, voxel_index(shape, i0, i1, i2));
      }
    }
  }
}

// Provisional check of a caller's tet_id (ROADMAP open question): −1, or an ok tet whose box holds
// the voxel, so that plane_values stays inside the int64 budget.
template <Real T>
bool valid_tet_id(const TetSetupView<T>& s, int32_t t, const int64_t (&x)[3]) {
  if (t == -1) {
    return true;
  }
  if (t < 0 || t >= s.num_tets || s.status[t] != TetStatus::ok) {
    return false;
  }
  for (int d = 0; d < 3; ++d) {
    if (x[d] < s.box_lo[t * 3 + d] || x[d] > s.box_hi[t * 3 + d]) {
      return false;
    }
  }
  return true;
}

template <Real T>
void store_barycentrics(const TetRecord<T>& rec, const int64_t (&x)[3], T* bary) {
  const Barycentrics<T> b = barycentric(rec, plane_values(rec, x));
  for (int j = 1; j < 4; ++j) {
    store_relaxed(bary[j - 1], b.pi[j]);
  }
}

}  // namespace

template <Real T>
void setup_tets(const MeshView<T>& mesh, int snap_bits, const TetSetupView<T>& out) {
  if (snap_bits < 0 || snap_bits > max_snap_bits) {
    throw std::invalid_argument("snap_bits must lie in [0, 24]");
  }
  if (mesh.num_vertices < 0) {
    throw std::invalid_argument("num_vertices is negative");
  }
  if (!(0 <= mesh.num_tets && mesh.num_tets <= INT32_MAX)) {
    throw std::invalid_argument("num_tets must lie in [0, 2^31 - 1]");
  }
  int64_t bad_indices = 0;
#pragma omp parallel for schedule(static) reduction(+ : bad_indices)
  for (int64_t i = 0; i < mesh.num_tets * 4; ++i) {
    const int32_t v = mesh.tetrahedra[i];
    bad_indices += (v < 0 || v >= mesh.num_vertices) ? 1 : 0;
  }
  if (bad_indices != 0) {
    throw std::invalid_argument("a tet index lies outside [0, num_vertices)");
  }

#pragma omp parallel for schedule(static)
  for (int64_t t = 0; t < mesh.num_tets; ++t) {
    double p[4][3];
    bool in_range = true;
    for (int v = 0; v < 4; ++v) {
      const int64_t vertex = mesh.tetrahedra[t * 4 + v];
      for (int d = 0; d < 3; ++d) {
        p[v][d] = static_cast<double>(mesh.positions[vertex * 3 + d]);
        in_range = in_range && in_snap_range(p[v][d], snap_bits);
      }
    }
    if (!in_range) {
      out.status[t] = TetStatus::outside_window;
      continue;
    }
    int64_t q[4][3];
    for (int v = 0; v < 4; ++v) {
      for (int d = 0; d < 3; ++d) {
        q[v][d] = snap(p[v][d], snap_bits);
      }
    }
    detail::store_record(out, t, setup_tet<T>(q, snap_bits));
  }
}

SkippedTets count_skipped(const TetStatus* status, int64_t num_tets) {
  int64_t inverted = 0;
  int64_t outside_window = 0;
#pragma omp parallel for schedule(static) reduction(+ : inverted, outside_window)
  for (int64_t t = 0; t < num_tets; ++t) {
    inverted += status[t] == TetStatus::inverted ? 1 : 0;
    outside_window += status[t] == TetStatus::outside_window ? 1 : 0;
  }
  return SkippedTets{inverted, outside_window};
}

template <Real T>
void locate(const TetSetupView<T>& s, GridShape shape, int32_t* tet_id) {
  check_shape(shape);
  fill(tet_id, num_voxels(shape), int32_t{-1});
#pragma omp parallel for schedule(dynamic, tet_chunk)
  for (int64_t t = 0; t < s.num_tets; ++t) {
    const TetRecord<T> rec = detail::load_record(s, t);
    if (rec.status != TetStatus::ok) {
      continue;
    }
    for_each_owned(rec, shape, [&](const int64_t (&)[3], int64_t index) {
      store_relaxed(tet_id[index], static_cast<int32_t>(t));
    });
  }
}

template <Real T>
void barycentrics(const TetSetupView<T>& s, GridShape shape, const int32_t* tet_id, T* bary) {
  check_shape(shape);
  int64_t bad_ids = 0;
#pragma omp parallel for schedule(static) reduction(+ : bad_ids)
  for (int64_t i0 = 0; i0 < shape[0]; ++i0) {
    for (int64_t i1 = 0; i1 < shape[1]; ++i1) {
      for (int64_t i2 = 0; i2 < shape[2]; ++i2) {
        const int64_t x[3] = {i0, i1, i2};
        bad_ids += valid_tet_id(s, tet_id[voxel_index(shape, i0, i1, i2)], x) ? 0 : 1;
      }
    }
  }
  if (bad_ids != 0) {
    throw std::invalid_argument(
        "tet_id holds a value outside [-1, num_tets), a skipped tet, or a tet whose box does not "
        "contain the voxel");
  }

#pragma omp parallel for schedule(static)
  for (int64_t i0 = 0; i0 < shape[0]; ++i0) {
    for (int64_t i1 = 0; i1 < shape[1]; ++i1) {
      for (int64_t i2 = 0; i2 < shape[2]; ++i2) {
        const int64_t index = voxel_index(shape, i0, i1, i2);
        const int32_t t = tet_id[index];
        if (t < 0) {
          for (int j = 0; j < 3; ++j) {
            bary[index * 3 + j] = T{0};
          }
          continue;
        }
        const int64_t x[3] = {i0, i1, i2};
        store_barycentrics(detail::load_record(s, t), x, bary + index * 3);
      }
    }
  }
}

template <Real T>
void rasterize(const TetSetupView<T>& s, GridShape shape, int32_t* tet_id, T* bary) {
  check_shape(shape);
  fill(tet_id, num_voxels(shape), int32_t{-1});
  fill(bary, num_voxels(shape) * 3, T{0});
#pragma omp parallel for schedule(dynamic, tet_chunk)
  for (int64_t t = 0; t < s.num_tets; ++t) {
    const TetRecord<T> rec = detail::load_record(s, t);
    if (rec.status != TetStatus::ok) {
      continue;
    }
    for_each_owned(rec, shape, [&](const int64_t (&x)[3], int64_t index) {
      store_relaxed(tet_id[index], static_cast<int32_t>(t));
      store_barycentrics(rec, x, bary + index * 3);
    });
  }
}

template <Real T>
void count_claims(const TetSetupView<T>& s, GridShape shape, int32_t* claims) {
  check_shape(shape);
  fill(claims, num_voxels(shape), int32_t{0});
#pragma omp parallel for schedule(dynamic, tet_chunk)
  for (int64_t t = 0; t < s.num_tets; ++t) {
    const TetRecord<T> rec = detail::load_record(s, t);
    if (rec.status != TetStatus::ok) {
      continue;
    }
    const Box box = clipped_box(rec, shape);
    for (int64_t i0 = box.lo[0]; i0 <= box.hi[0]; ++i0) {
      for (int64_t i1 = box.lo[1]; i1 <= box.hi[1]; ++i1) {
        for (int64_t i2 = box.lo[2]; i2 <= box.hi[2]; ++i2) {
          const int64_t x[3] = {i0, i1, i2};
          if (owns(plane_values(rec, x))) {
            std::atomic_ref<int32_t>(claims[voxel_index(shape, i0, i1, i2)])
                .fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
    }
  }
}

#define TETVOX_INSTANTIATE_CPU(T)                                                       \
  template void setup_tets<T>(const MeshView<T>&, int, const TetSetupView<T>&);         \
  template void locate<T>(const TetSetupView<T>&, GridShape, int32_t*);                 \
  template void barycentrics<T>(const TetSetupView<T>&, GridShape, const int32_t*, T*); \
  template void rasterize<T>(const TetSetupView<T>&, GridShape, int32_t*, T*);          \
  template void count_claims<T>(const TetSetupView<T>&, GridShape, int32_t*);
TETVOX_INSTANTIATE_CPU(float)
TETVOX_INSTANTIATE_CPU(double)
#undef TETVOX_INSTANTIATE_CPU

}  // namespace tetvox::cpu
