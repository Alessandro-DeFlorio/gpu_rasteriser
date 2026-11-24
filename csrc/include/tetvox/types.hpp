// Public types of the tetvox ops (ADR 0009): dtype constraint, grid shape, per-tet status and the
// caller-owned views on the mesh and on the per-tet setup record.
#pragma once

#include <array>
#include <concepts>
#include <cstdint>

// Marks the shared procedures callable from host and device code (ADR 0006).
#if defined(__CUDACC__)
#define TETVOX_HD __host__ __device__
#else
#define TETVOX_HD
#endif

namespace tetvox {

// The real dtypes every op is instantiated for (ADR 0008).
template <class T>
concept Real = std::same_as<T, float> || std::same_as<T, double>;

// Voxel grid shape. Axes 0, 1, 2; C order, axis 2 fastest (ADR 0007).
using GridShape = std::array<int64_t, 3>;

// k of ADR 0006: vertices are snapped to multiples of 2^-k voxel.
inline constexpr int default_snap_bits = 14;
// Largest accepted k (ADR 0009): keeps f_j = r / 2^k exact in float32 (24-bit significand).
inline constexpr int max_snap_bits = 24;
// Bit budget of the int64 per-voxel test: 2k + 3L <= 59 (ADR 0006, derivation in R1 §2.2).
inline constexpr int int64_budget_bits = 59;
// |v| <= 2^(62 - k) voxels, so that v * 2^k fits int64 (ADR 0008).
inline constexpr int snap_range_bits = 62;

// L of ADR 0006 for a given k: the largest window exponent with 2k + 3L <= 59 (ADR 0008).
TETVOX_HD constexpr int window_bits(int snap_bits) {
  return (int64_budget_bits - 2 * snap_bits) / 3;
}

// Classification of a tet by setup_tets, in the order of ADR 0008.
enum class TetStatus : uint8_t {
  ok = 0,              // rasterised (clipped to the grid, possibly empty)
  inverted = 1,        // D <= 0: inverted or degenerate after snapping
  outside_window = 2,  // non-finite coordinate, |v| > 2^(62-k), or a span above 2^(k+L)
};

// Number of tets of each skipped kind (count_skipped).
struct SkippedTets {
  int64_t inverted;
  int64_t outside_window;
};

// Caller-owned, read-only mesh. positions (num_vertices, 3), tetrahedra (num_tets, 4), C order.
template <Real T>
struct MeshView {
  const T* positions;
  const int32_t* tetrahedra;
  int64_t num_vertices;
  int64_t num_tets;
};

// Caller-owned per-tet setup record (structure of arrays, ADR 0009), written only by setup_tets.
// For a tet whose status is not ok, every field other than status is unspecified.
template <Real T>
struct TetSetupView {
  int64_t num_tets;
  int64_t* normals;   // (num_tets, 4, 3) inward face normals n_j, lattice units
  int64_t* offsets;   // (num_tets, 4) c_j, tie-break folded in
  T* fractions;       // (num_tets, 4) f_j in [0, 1]
  T* bary_scale;      // (num_tets,) 2^k / D
  int64_t* box_lo;    // (num_tets, 3) O = ceil(min q / 2^k), not clipped to the grid
  int64_t* box_hi;    // (num_tets, 3) floor(max q / 2^k)
  TetStatus* status;  // (num_tets,)
};

}  // namespace tetvox
