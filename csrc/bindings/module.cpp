// tetvox._core: the nanobind module (ADR 0001, 0009, 0010). One overload per dtype (and, later,
// backend); nanobind overload resolution dispatches on dtype, device and number of array
// dimensions. Arrays are never converted or copied (noconvert): a wrong dtype, layout, device or
// ndim is a TypeError (ADR 0008). Each binding reads the simplex dimension D from the array shapes,
// calls the matching instantiation and checks that the shapes agree with D (ValueError otherwise);
// everything else is validated by the C++ ops.
//
// A SimplexSetup crosses the boundary as its seven arrays, in field order (ADR 0009).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "tetvox/config.hpp"
#include "tetvox/cpu.hpp"
#include "tetvox/device.hpp"

namespace nb = nanobind;

namespace {

template <class V, std::size_t N>
using CpuArray = nb::ndarray<V, nb::ndim<N>, nb::c_contig, nb::device::cpu>;
// A grid of any dimension; its ndim is checked against D.
template <class V>
using CpuGrid = nb::ndarray<V, nb::c_contig, nb::device::cpu>;

// SimplexSetup fields; C is const for the consumers, which only read the record.
template <class C>
using Normals = CpuArray<C, 3>;  // (T, D + 1, D)
template <class C>
using Offsets = CpuArray<C, 2>;  // (T, D + 1)
template <class C>
using Fractions = CpuArray<C, 2>;  // (T, D + 1)
template <class C>
using BaryScale = CpuArray<C, 1>;  // (T,)
template <class C>
using Box = CpuArray<C, 2>;  // (T, D)
template <class C>
using Status = CpuArray<C, 1>;  // (T,)

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

// Calls f.template operator()<D>() with the compile-time D equal to dim.
template <std::size_t D = 1, class F>
void with_dim(std::size_t dim, F&& f) {
  if constexpr (D <= tetvox::max_dim) {
    if (dim == D) {
      f.template operator()<D>();
      return;
    }
    with_dim<D + 1>(dim, std::forward<F>(f));
  } else {
    throw std::invalid_argument("the simplex dimension D must be 1, 2 or 3");
  }
}

// The view on a SimplexSetup, after checking that its arrays agree on the number of simplices and
// on D.
template <class T, std::size_t D, class I64, class R, class U8>
tetvox::SimplexSetupView<T, D> make_view(const Normals<I64>& normals, const Offsets<I64>& offsets,
                                         const Fractions<R>& fractions,
                                         const BaryScale<R>& bary_scale, const Box<I64>& box_lo,
                                         const Box<I64>& box_hi, const Status<U8>& status) {
  const size_t n = status.shape(0);
  require(normals.shape(0) == n && offsets.shape(0) == n && fractions.shape(0) == n &&
              bary_scale.shape(0) == n && box_lo.shape(0) == n && box_hi.shape(0) == n,
          "the SimplexSetup arrays disagree on the number of simplices");
  require(normals.shape(1) == D + 1 && normals.shape(2) == D && offsets.shape(1) == D + 1 &&
              fractions.shape(1) == D + 1 && box_lo.shape(1) == D && box_hi.shape(1) == D,
          "the SimplexSetup arrays disagree on the dimension D");
  return tetvox::SimplexSetupView<T, D>{
      static_cast<int64_t>(n),
      const_cast<int64_t*>(normals.data()),
      const_cast<int64_t*>(offsets.data()),
      const_cast<T*>(fractions.data()),
      const_cast<T*>(bary_scale.data()),
      const_cast<int64_t*>(box_lo.data()),
      const_cast<int64_t*>(box_hi.data()),
      reinterpret_cast<tetvox::SimplexStatus*>(const_cast<uint8_t*>(status.data())),
  };
}

// The grid shape of a D-dimensional grid array.
template <std::size_t D, class A>
tetvox::GridShape<D> grid_shape(const A& grid, const char* name) {
  require(grid.ndim() == D, name);
  tetvox::GridShape<D> shape{};
  for (std::size_t d = 0; d < D; ++d) {
    shape[d] = static_cast<int64_t>(grid.shape(d));
  }
  return shape;
}

// bary must have the shape (*grid, D).
template <std::size_t D, class B>
void require_bary_shape(const tetvox::GridShape<D>& grid, const B& bary) {
  bool same = bary.ndim() == D + 1 && bary.shape(D) == D;
  for (std::size_t d = 0; same && d < D; ++d) {
    same = static_cast<int64_t>(bary.shape(d)) == grid[d];
  }
  require(same, "bary must have the shape (*simplex_id.shape, D)");
}

auto setup_args() {
  return std::make_tuple(nb::arg("normals").noconvert(), nb::arg("offsets").noconvert(),
                         nb::arg("fractions").noconvert(), nb::arg("bary_scale").noconvert(),
                         nb::arg("box_lo").noconvert(), nb::arg("box_hi").noconvert(),
                         nb::arg("status").noconvert());
}

// m.def with `before`, the seven SimplexSetup arguments, then `after`; the GIL is released.
template <class F, class... Before, class... After>
void def_op(nb::module_& m, const char* name, F f, std::tuple<Before...> before,
            std::tuple<After...> after) {
  std::apply(
      [&](auto&&... args) { m.def(name, f, args..., nb::call_guard<nb::gil_scoped_release>()); },
      std::tuple_cat(std::move(before), setup_args(), std::move(after)));
}

constexpr const char* grid_ndim_message = "the grid array must have D dimensions";

template <class T>
void bind_cpu(nb::module_& m) {
  using I64 = const int64_t;
  using R = const T;
  using U8 = const uint8_t;

  def_op(
      m, "setup_simplices",
      [](CpuArray<const T, 2> positions, CpuArray<const int32_t, 2> simplices, int snap_bits,
         Normals<int64_t> normals, Offsets<int64_t> offsets, Fractions<T> fractions,
         BaryScale<T> bary_scale, Box<int64_t> box_lo, Box<int64_t> box_hi,
         Status<uint8_t> status) {
        with_dim(positions.shape(1), [&]<std::size_t D>() {
          require(simplices.shape(1) == D + 1, "simplices must have D + 1 columns");
          const auto out =
              make_view<T, D>(normals, offsets, fractions, bary_scale, box_lo, box_hi, status);
          require(out.num_simplices == static_cast<int64_t>(simplices.shape(0)),
                  "the SimplexSetup arrays and simplices disagree on the number of simplices");
          const tetvox::MeshView<T, D> mesh{positions.data(), simplices.data(),
                                            static_cast<int64_t>(positions.shape(0)),
                                            out.num_simplices};
          tetvox::cpu::setup_simplices(mesh, snap_bits, out);
        });
      },
      std::make_tuple(nb::arg("positions").noconvert(), nb::arg("simplices").noconvert(),
                      nb::arg("snap_bits")),
      std::tuple<>{});

  def_op(
      m, "locate",
      [](Normals<I64> normals, Offsets<I64> offsets, Fractions<R> fractions,
         BaryScale<R> bary_scale, Box<I64> box_lo, Box<I64> box_hi, Status<U8> status,
         CpuGrid<int32_t> simplex_id) {
        with_dim(box_lo.shape(1), [&]<std::size_t D>() {
          const auto s =
              make_view<T, D>(normals, offsets, fractions, bary_scale, box_lo, box_hi, status);
          tetvox::cpu::locate(s, grid_shape<D>(simplex_id, grid_ndim_message), simplex_id.data());
        });
      },
      std::tuple<>{}, std::make_tuple(nb::arg("simplex_id").noconvert()));

  def_op(
      m, "barycentrics",
      [](Normals<I64> normals, Offsets<I64> offsets, Fractions<R> fractions,
         BaryScale<R> bary_scale, Box<I64> box_lo, Box<I64> box_hi, Status<U8> status,
         CpuGrid<const int32_t> simplex_id, CpuGrid<T> bary) {
        with_dim(box_lo.shape(1), [&]<std::size_t D>() {
          const auto s =
              make_view<T, D>(normals, offsets, fractions, bary_scale, box_lo, box_hi, status);
          const auto shape = grid_shape<D>(simplex_id, grid_ndim_message);
          require_bary_shape(shape, bary);
          tetvox::cpu::barycentrics(s, shape, simplex_id.data(), bary.data());
        });
      },
      std::tuple<>{},
      std::make_tuple(nb::arg("simplex_id").noconvert(), nb::arg("bary").noconvert()));

  def_op(
      m, "rasterize",
      [](Normals<I64> normals, Offsets<I64> offsets, Fractions<R> fractions,
         BaryScale<R> bary_scale, Box<I64> box_lo, Box<I64> box_hi, Status<U8> status,
         CpuGrid<int32_t> simplex_id, CpuGrid<T> bary) {
        with_dim(box_lo.shape(1), [&]<std::size_t D>() {
          const auto s =
              make_view<T, D>(normals, offsets, fractions, bary_scale, box_lo, box_hi, status);
          const auto shape = grid_shape<D>(simplex_id, grid_ndim_message);
          require_bary_shape(shape, bary);
          tetvox::cpu::rasterize(s, shape, simplex_id.data(), bary.data());
        });
      },
      std::tuple<>{},
      std::make_tuple(nb::arg("simplex_id").noconvert(), nb::arg("bary").noconvert()));

  def_op(
      m, "count_claims",
      [](Normals<I64> normals, Offsets<I64> offsets, Fractions<R> fractions,
         BaryScale<R> bary_scale, Box<I64> box_lo, Box<I64> box_hi, Status<U8> status,
         CpuGrid<int32_t> claims) {
        with_dim(box_lo.shape(1), [&]<std::size_t D>() {
          const auto s =
              make_view<T, D>(normals, offsets, fractions, bary_scale, box_lo, box_hi, status);
          tetvox::cpu::count_claims(s, grid_shape<D>(claims, grid_ndim_message), claims.data());
        });
      },
      std::tuple<>{}, std::make_tuple(nb::arg("claims").noconvert()));
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "tetvox C++/CUDA core";
  m.attr("cuda_enabled") = tetvox::cuda_enabled;
  m.attr("default_snap_bits") = tetvox::default_snap_bits;
  m.attr("max_snap_bits") = tetvox::max_snap_bits;
  m.def("device_count", &tetvox::device_count,
        "Number of visible CUDA devices (always 0 in a CPU-only build).");
  m.def(
      "count_skipped",
      [](Status<const uint8_t> status) {
        const tetvox::SkippedSimplices s = tetvox::cpu::count_skipped(
            reinterpret_cast<const tetvox::SimplexStatus*>(status.data()),
            static_cast<int64_t>(status.shape(0)));
        return std::make_tuple(s.inverted, s.outside_window);
      },
      nb::arg("status").noconvert(), nb::call_guard<nb::gil_scoped_release>());
  bind_cpu<float>(m);
  bind_cpu<double>(m);
}
