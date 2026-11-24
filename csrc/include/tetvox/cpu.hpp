// CPU backend of the rasterisation ops (ADR 0007, 0009). Same contract as tetvox::cuda, and the
// same per-element procedures (tetvox/raster.hpp), so the results are bit-identical.
//
// Outputs are caller-allocated; each op writes every element of its outputs. Grids are C order,
// axis 2 fastest: voxel (i0, i1, i2) is element (i0 · shape[1] + i1) · shape[2] + i2. A
// TetSetupView passed to a consumer is trusted to come from setup_tets on the same mesh.
// Invalid arguments throw std::invalid_argument before any output is written (ADR 0008).
#pragma once

#include <cstdint>

#include "tetvox/types.hpp"

namespace tetvox::cpu {

// Snaps and classifies every tet and writes its record (out.num_tets == mesh.num_tets). Throws if
// snap_bits is outside [0, max_snap_bits], num_vertices < 0, num_tets is outside [0, INT32_MAX],
// or a tet index is outside [0, num_vertices).
template <Real T>
void setup_tets(const MeshView<T>& mesh, int snap_bits, const TetSetupView<T>& out);

// Number of tets of each skipped status.
SkippedTets count_skipped(const TetStatus* status, int64_t num_tets);

// tet_id (shape): the tet owning each voxel centre, −1 where none does. Throws on a negative shape.
template <Real T>
void locate(const TetSetupView<T>& s, GridShape shape, int32_t* tet_id);

// bary (shape, 3): π_1..π_3 of each voxel centre in the tet tet_id names, 0 where tet_id is −1.
// Throws on a negative shape, or (provisional, ROADMAP) a tet_id outside [−1, num_tets), naming a
// tet whose status is not ok, or naming a tet whose box does not contain the voxel.
template <Real T>
void barycentrics(const TetSetupView<T>& s, GridShape shape, const int32_t* tet_id, T* bary);

// locate and barycentrics in one pass; equal to them bit for bit.
template <Real T>
void rasterize(const TetSetupView<T>& s, GridShape shape, int32_t* tet_id, T* bary);

// claims (shape): the number of tets whose inside test holds at each voxel centre (diagnostic: 1
// everywhere a mesh tiles the grid). Zeroed by the call. Throws on a negative shape.
template <Real T>
void count_claims(const TetSetupView<T>& s, GridShape shape, int32_t* claims);

}  // namespace tetvox::cpu
