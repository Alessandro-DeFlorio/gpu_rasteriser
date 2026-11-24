// tetvox._core: the nanobind module (ADR 0001, 0009). One overload per dtype (and, later, backend);
// nanobind overload resolution dispatches on dtype and device. Arrays are never converted or copied
// (noconvert): a wrong dtype, layout or device is a TypeError (ADR 0008). The binding checks that
// array shapes agree; everything else is validated by the C++ ops.
//
// A TetSetup crosses the boundary as its seven arrays, in field order (ADR 0009).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "tetvox/config.hpp"
#include "tetvox/cpu.hpp"
#include "tetvox/device.hpp"

namespace nb = nanobind;

namespace {

template <class V, ssize_t... Dims>
using CpuArray = nb::ndarray<V, nb::shape<Dims...>, nb::c_contig, nb::device::cpu>;

// TetSetup fields; C is const for the consumers, which only read the record.
template <class C>
using Normals = CpuArray<C, -1, 4, 3>;
template <class C>
using Offsets = CpuArray<C, -1, 4>;
template <class C>
using Fractions = CpuArray<C, -1, 4>;
template <class C>
using BaryScale = CpuArray<C, -1>;
template <class C>
using Box = CpuArray<C, -1, 3>;
template <class C>
using Status = CpuArray<C, -1>;

template <class C>
using Grid = CpuArray<C, -1, -1, -1>;
template <class C>
using GridVec3 = CpuArray<C, -1, -1, -1, 3>;

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

// The view on a TetSetup, after checking that its arrays agree on the number of tets.
template <class T, class I64, class R, class U8>
tetvox::TetSetupView<T> make_view(const Normals<I64>& normals, const Offsets<I64>& offsets,
                                  const Fractions<R>& fractions, const BaryScale<R>& bary_scale,
                                  const Box<I64>& box_lo, const Box<I64>& box_hi,
                                  const Status<U8>& status) {
  const size_t n = status.shape(0);
  require(normals.shape(0) == n && offsets.shape(0) == n && fractions.shape(0) == n &&
              bary_scale.shape(0) == n && box_lo.shape(0) == n && box_hi.shape(0) == n,
          "the TetSetup arrays disagree on the number of tets");
  return tetvox::TetSetupView<T>{
      static_cast<int64_t>(n),
      const_cast<int64_t*>(normals.data()),
      const_cast<int64_t*>(offsets.data()),
      const_cast<T*>(fractions.data()),
      const_cast<T*>(bary_scale.data()),
      const_cast<int64_t*>(box_lo.data()),
      const_cast<int64_t*>(box_hi.data()),
      reinterpret_cast<tetvox::TetStatus*>(const_cast<uint8_t*>(status.data())),
  };
}

template <class A>
tetvox::GridShape grid_shape(const A& grid) {
  return {static_cast<int64_t>(grid.shape(0)), static_cast<int64_t>(grid.shape(1)),
          static_cast<int64_t>(grid.shape(2))};
}

template <class A, class B>
void require_same_grid(const A& tet_id, const B& bary) {
  require(tet_id.shape(0) == bary.shape(0) && tet_id.shape(1) == bary.shape(1) &&
              tet_id.shape(2) == bary.shape(2),
          "tet_id and bary disagree on the grid shape");
}

auto setup_args() {
  return std::make_tuple(nb::arg("normals").noconvert(), nb::arg("offsets").noconvert(),
                         nb::arg("fractions").noconvert(), nb::arg("bary_scale").noconvert(),
                         nb::arg("box_lo").noconvert(), nb::arg("box_hi").noconvert(),
                         nb::arg("status").noconvert());
}

// m.def with `before`, the seven TetSetup arguments, then `after`; the GIL is released.
template <class F, class... Before, class... After>
void def_op(nb::module_& m, const char* name, F f, std::tuple<Before...> before,
            std::tuple<After...> after) {
  std::apply(
      [&](auto&&... args) { m.def(name, f, args..., nb::call_guard<nb::gil_scoped_release>()); },
      std::tuple_cat(std::move(before), setup_args(), std::move(after)));
}

template <class T>
void bind_cpu(nb::module_& m) {
  using I64 = const int64_t;
  using R = const T;
  using U8 = const uint8_t;

  def_op(
      m, "setup_tets",
      [](CpuArray<const T, -1, 3> positions, CpuArray<const int32_t, -1, 4> tetrahedra,
         int snap_bits, Normals<int64_t> normals, Offsets<int64_t> offsets, Fractions<T> fractions,
         BaryScale<T> bary_scale, Box<int64_t> box_lo, Box<int64_t> box_hi,
         Status<uint8_t> status) {
        const auto out =
            make_view<T>(normals, offsets, fractions, bary_scale, box_lo, box_hi, status);
        require(out.num_tets == static_cast<int64_t>(tetrahedra.shape(0)),
                "the TetSetup arrays and tetrahedra disagree on the number of tets");
        const tetvox::MeshView<T> mesh{positions.data(), tetrahedra.data(),
                                       static_cast<int64_t>(positions.shape(0)), out.num_tets};
        tetvox::cpu::setup_tets(mesh, snap_bits, out);
      },
      std::make_tuple(nb::arg("positions").noconvert(), nb::arg("tetrahedra").noconvert(),
                      nb::arg("snap_bits")),
      std::tuple<>{});

  def_op(
      m, "locate",
      [](Normals<I64> normals, Offsets<I64> offsets, Fractions<R> fractions,
         BaryScale<R> bary_scale, Box<I64> box_lo, Box<I64> box_hi, Status<U8> status,
         Grid<int32_t> tet_id) {
        const auto s =
            make_view<T>(normals, offsets, fractions, bary_scale, box_lo, box_hi, status);
        tetvox::cpu::locate(s, grid_shape(tet_id), tet_id.data());
      },
      std::tuple<>{}, std::make_tuple(nb::arg("tet_id").noconvert()));

  def_op(
      m, "barycentrics",
      [](Normals<I64> normals, Offsets<I64> offsets, Fractions<R> fractions,
         BaryScale<R> bary_scale, Box<I64> box_lo, Box<I64> box_hi, Status<U8> status,
         Grid<const int32_t> tet_id, GridVec3<T> bary) {
        const auto s =
            make_view<T>(normals, offsets, fractions, bary_scale, box_lo, box_hi, status);
        require_same_grid(tet_id, bary);
        tetvox::cpu::barycentrics(s, grid_shape(tet_id), tet_id.data(), bary.data());
      },
      std::tuple<>{}, std::make_tuple(nb::arg("tet_id").noconvert(), nb::arg("bary").noconvert()));

  def_op(
      m, "rasterize",
      [](Normals<I64> normals, Offsets<I64> offsets, Fractions<R> fractions,
         BaryScale<R> bary_scale, Box<I64> box_lo, Box<I64> box_hi, Status<U8> status,
         Grid<int32_t> tet_id, GridVec3<T> bary) {
        const auto s =
            make_view<T>(normals, offsets, fractions, bary_scale, box_lo, box_hi, status);
        require_same_grid(tet_id, bary);
        tetvox::cpu::rasterize(s, grid_shape(tet_id), tet_id.data(), bary.data());
      },
      std::tuple<>{}, std::make_tuple(nb::arg("tet_id").noconvert(), nb::arg("bary").noconvert()));

  def_op(
      m, "count_claims",
      [](Normals<I64> normals, Offsets<I64> offsets, Fractions<R> fractions,
         BaryScale<R> bary_scale, Box<I64> box_lo, Box<I64> box_hi, Status<U8> status,
         Grid<int32_t> claims) {
        const auto s =
            make_view<T>(normals, offsets, fractions, bary_scale, box_lo, box_hi, status);
        tetvox::cpu::count_claims(s, grid_shape(claims), claims.data());
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
        const tetvox::SkippedTets s =
            tetvox::cpu::count_skipped(reinterpret_cast<const tetvox::TetStatus*>(status.data()),
                                       static_cast<int64_t>(status.shape(0)));
        return std::make_tuple(s.inverted, s.outside_window);
      },
      nb::arg("status").noconvert(), nb::call_guard<nb::gil_scoped_release>());
  bind_cpu<float>(m);
  bind_cpu<double>(m);
}
