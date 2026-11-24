// CPU backend of the rasterisation ops (ADR 0007, 0009, 0010). Same contract as tetvox::cuda, and
// the same per-element procedures (tetvox/raster.hpp), so the results are bit-identical.
//
// Every op is instantiated for float and double and for D = 1, 2, 3 (segments, triangles,
// tetrahedra). Outputs are caller-allocated; each op writes every element of its outputs. Grids are
// C order, axis D−1 fastest: voxel (i_0, …, i_{D−1}) is element
// (…(i_0 · shape[1] + i_1) · shape[2] + …) + i_{D−1}. A SimplexSetupView passed to a consumer is
// trusted to come from setup_simplices on the same mesh. Invalid arguments throw
// std::invalid_argument before any output is written (ADR 0008).
#pragma once

#include <cstddef>
#include <cstdint>

#include "tetvox/types.hpp"

namespace tetvox::cpu {

// Snaps and classifies every simplex and writes its record (out.num_simplices ==
// mesh.num_simplices). Throws if snap_bits is outside [0, max_snap_bits], num_vertices < 0,
// num_simplices is outside [0, INT32_MAX], or a simplex index is outside [0, num_vertices).
template <Real T, std::size_t D>
void setup_simplices(const MeshView<T, D>& mesh, int snap_bits, const SimplexSetupView<T, D>& out);

// Number of simplices of each skipped status.
SkippedSimplices count_skipped(const SimplexStatus* status, int64_t num_simplices);

// simplex_id (shape): the simplex owning each voxel centre, −1 where none does. Throws on a
// negative shape.
template <Real T, std::size_t D>
void locate(const SimplexSetupView<T, D>& s, GridShape<D> shape, int32_t* simplex_id);

// bary (shape, D): π_1..π_D of each voxel centre in the simplex simplex_id names, 0 where
// simplex_id is −1. Throws on a negative shape, or (provisional, ROADMAP) a simplex_id outside
// [−1, num_simplices), naming a simplex whose status is not ok, or naming a simplex whose box does
// not contain the voxel.
template <Real T, std::size_t D>
void barycentrics(const SimplexSetupView<T, D>& s, GridShape<D> shape, const int32_t* simplex_id,
                  T* bary);

// locate and barycentrics in one pass; equal to them bit for bit.
template <Real T, std::size_t D>
void rasterize(const SimplexSetupView<T, D>& s, GridShape<D> shape, int32_t* simplex_id, T* bary);

// claims (shape): the number of simplices whose inside test holds at each voxel centre
// (diagnostic: 1 everywhere a mesh tiles the grid). Zeroed by the call. Throws on a negative shape.
template <Real T, std::size_t D>
void count_claims(const SimplexSetupView<T, D>& s, GridShape<D> shape, int32_t* claims);

}  // namespace tetvox::cpu
