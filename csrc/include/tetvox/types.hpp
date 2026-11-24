// Public types of the tetvox ops (ADR 0009, 0010): dtype and dimension constraints, grid shape,
// per-simplex status and the caller-owned views on the mesh and on the per-simplex setup record.
#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>

// Marks the shared procedures callable from host and device code (ADR 0006).
#if defined(__CUDACC__)
#define TETVOX_HD __host__ __device__
#else
#define TETVOX_HD
#endif

// Forces inlining of a hot procedure (ADR 0010: row_span, about 5 % on the CPU in R3 §5.2).
#if defined(__CUDACC__)
#define TETVOX_FORCEINLINE __forceinline__
#elif defined(__GNUC__)
#define TETVOX_FORCEINLINE [[gnu::always_inline]] inline
#else
#define TETVOX_FORCEINLINE inline
#endif

namespace tetvox {

// The real dtypes every op is instantiated for (ADR 0008).
template <class T>
concept Real = std::same_as<T, float> || std::same_as<T, double>;

// Largest supported dimension (ADR 0010): segments, triangles and tetrahedra.
inline constexpr std::size_t max_dim = 3;

// The dimensions every op is instantiated for (ADR 0010).
template <std::size_t D>
concept SupportedDim = D >= 1 && D <= max_dim;

// Voxel grid shape. Axes 0 … D−1; C order, axis D−1 fastest (ADR 0007, 0010).
template <std::size_t D>
using GridShape = std::array<int64_t, D>;

// k of ADR 0006: vertices are snapped to multiples of 2^-k voxel.
inline constexpr int default_snap_bits = 14;
// Largest accepted k (ADR 0009): keeps f_j = r / 2^k exact in float32 (24-bit significand).
inline constexpr int max_snap_bits = 24;
// |v| <= 2^(62 - k) voxels, so that v * 2^k fits int64 (ADR 0008).
inline constexpr int snap_range_bits = 62;

namespace detail {

constexpr int floor_log2(std::size_t x) { return x <= 1 ? 0 : 1 + floor_log2(x >> 1); }
constexpr std::size_t factorial(std::size_t n) { return n <= 1 ? 1 : n * factorial(n - 1); }

}  // namespace detail

// Bit budget of the int64 per-voxel test: (D−1)·k + D·L <= 62 − floor(log2(2·D!)), that is 61, 60
// and 59 for D = 1, 2, 3. The per-sample bound |t_j| <= 2·D!·2^((D−1)k + DL) + 1 must stay below
// 2^63 (ADR 0010; derivation in R3 §2.1; the D = 3 case is ADR 0006's 2k + 3L <= 59).
template <std::size_t D>
  requires SupportedDim<D>
inline constexpr int int64_budget_bits = 62 - detail::floor_log2(2 * detail::factorial(D));

// L for a given k: the largest window exponent within the budget, and with k + L <= 63 so that
// the window 2^(k+L) fits the unsigned window test (binds in 1D only; ADR 0010).
template <std::size_t D>
  requires SupportedDim<D>
TETVOX_HD constexpr int window_bits(int snap_bits) {
  const int dim = static_cast<int>(D);
  const int budget = (int64_budget_bits<D> - (dim - 1) * snap_bits) / dim;
  const int representable = 63 - snap_bits;
  return budget < representable ? budget : representable;
}

// Classification of a simplex by setup_simplices, in the order of ADR 0008.
enum class SimplexStatus : uint8_t {
  ok = 0,              // rasterised (clipped to the grid, possibly empty)
  inverted = 1,        // D <= 0: inverted or degenerate after snapping
  outside_window = 2,  // non-finite coordinate, |v| > 2^(62-k), or a span above 2^(k+L)
};

// Number of simplices of each skipped kind (count_skipped).
struct SkippedSimplices {
  int64_t inverted;
  int64_t outside_window;
};

// Caller-owned, read-only mesh. positions (num_vertices, D), simplices (num_simplices, D + 1),
// C order.
template <Real T, std::size_t D>
  requires SupportedDim<D>
struct MeshView {
  const T* positions;
  const int32_t* simplices;
  int64_t num_vertices;
  int64_t num_simplices;
};

// Caller-owned per-simplex setup record (structure of arrays, ADR 0009), written only by
// setup_simplices. For a simplex whose status is not ok, every field other than status is
// unspecified.
template <Real T, std::size_t D>
  requires SupportedDim<D>
struct SimplexSetupView {
  int64_t num_simplices;
  int64_t* normals;       // (num_simplices, D + 1, D) inward face normals n_j, lattice units
  int64_t* offsets;       // (num_simplices, D + 1) c_j, tie-break folded in
  T* fractions;           // (num_simplices, D + 1) f_j in [0, 1]
  T* bary_scale;          // (num_simplices,) 2^k / D
  int64_t* box_lo;        // (num_simplices, D) O = ceil(min q / 2^k), not clipped to the grid
  int64_t* box_hi;        // (num_simplices, D) floor(max q / 2^k)
  SimplexStatus* status;  // (num_simplices,)
};

}  // namespace tetvox
