// Minimal consumer of the installed tetvox C++ package (ADR 0004): reports the build, then
// rasterises a Kuhn cube in 1D, 2D and 3D on the CPU and checks that it owns its voxels exactly
// once (ADR 0010).
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "tetvox/config.hpp"
#include "tetvox/cpu.hpp"
#include "tetvox/device.hpp"

namespace {

// Side of the cube [0, side]^D, in voxels.
constexpr int64_t side = 3;

// The D! simplices of the Kuhn triangulation of a cube: for each order of the axes, the path from
// vertex 0 to vertex 2^D − 1 adding one axis bit at a time. Vertex i sits at side · (bits of i).
// Simplices with a negative determinant get their first two vertices swapped.
template <std::size_t D>
std::vector<int32_t> kuhn_simplices() {
  std::array<int, D> order{};
  for (std::size_t d = 0; d < D; ++d) {
    order[d] = static_cast<int>(d);
  }
  std::vector<int32_t> simplices;
  do {
    std::array<int32_t, D + 1> s{};
    for (std::size_t v = 1; v <= D; ++v) {
      s[v] = s[v - 1] | (1 << order[v - 1]);
    }
    // The edges are side · (e_order[0], e_order[0] + e_order[1], …), so the determinant has the
    // sign of the permutation `order`.
    int inversions = 0;
    for (std::size_t a = 0; a < D; ++a) {
      for (std::size_t b = a + 1; b < D; ++b) {
        inversions += order[a] > order[b] ? 1 : 0;
      }
    }
    if (inversions % 2 == 1) {
      std::swap(s[0], s[1]);
    }
    simplices.insert(simplices.end(), s.begin(), s.end());
  } while (std::next_permutation(order.begin(), order.end()));
  return simplices;
}

template <std::size_t D>
bool rasterize_kuhn_cube() {
  const int64_t num_vertices = int64_t{1} << D;
  std::vector<float> positions;
  for (int64_t i = 0; i < num_vertices; ++i) {
    for (std::size_t d = 0; d < D; ++d) {
      positions.push_back(static_cast<float>(side * ((i >> d) & 1)));
    }
  }
  const std::vector<int32_t> simplices = kuhn_simplices<D>();
  const auto num = static_cast<int64_t>(simplices.size() / (D + 1));

  std::vector<int64_t> normals(num * (D + 1) * D), offsets(num * (D + 1)), box_lo(num * D),
      box_hi(num * D);
  std::vector<float> fractions(num * (D + 1)), bary_scale(num);
  std::vector<tetvox::SimplexStatus> status(num);
  const tetvox::SimplexSetupView<float, D> setup{
      num,           normals.data(), offsets.data(), fractions.data(), bary_scale.data(),
      box_lo.data(), box_hi.data(),  status.data()};
  const tetvox::MeshView<float, D> mesh{positions.data(), simplices.data(), num_vertices, num};
  tetvox::cpu::setup_simplices(mesh, tetvox::default_snap_bits, setup);
  const tetvox::SkippedSimplices skipped = tetvox::cpu::count_skipped(status.data(), num);

  tetvox::GridShape<D> shape{};
  int64_t voxels = 1;
  for (std::size_t d = 0; d < D; ++d) {
    shape[d] = side + 1;
    voxels *= side + 1;
  }
  std::vector<int32_t> simplex_id(voxels), claims(voxels);
  std::vector<float> bary(voxels * D);
  tetvox::cpu::rasterize(setup, shape, simplex_id.data(), bary.data());
  tetvox::cpu::count_claims(setup, shape, claims.data());

  // The half-open box [0, side)^D is owned exactly once; the layers at index side are not owned.
  int64_t errors = 0;
  for (int64_t i = 0; i < voxels; ++i) {
    bool inside = true;
    for (int64_t rest = i; rest > 0; rest /= side + 1) {
      inside = inside && rest % (side + 1) < side;
    }
    errors += (claims[i] != (inside ? 1 : 0)) || ((simplex_id[i] >= 0) != inside);
  }
  std::printf("tetvox: %zuD Kuhn cube, %lld inverted, %lld outside window, %lld voxel errors\n", D,
              static_cast<long long>(skipped.inverted),
              static_cast<long long>(skipped.outside_window), static_cast<long long>(errors));
  return skipped.inverted == 0 && skipped.outside_window == 0 && errors == 0;
}

}  // namespace

int main() {
  const int devices = tetvox::device_count();
  std::printf("tetvox (CUDA %s): %d CUDA device(s) visible\n",
              tetvox::cuda_enabled ? "enabled" : "disabled", devices);
  // A CUDA build is checked on a machine with a GPU; a CPU-only build must report none.
  const bool devices_ok = tetvox::cuda_enabled ? devices >= 1 : devices == 0;
  const bool cubes_ok =
      rasterize_kuhn_cube<1>() && rasterize_kuhn_cube<2>() && rasterize_kuhn_cube<3>();
  return devices_ok && cubes_ok ? 0 : 1;
}
