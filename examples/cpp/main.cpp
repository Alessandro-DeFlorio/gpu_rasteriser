// Minimal consumer of the installed tetvox C++ package (ADR 0004): reports the build, then
// rasterises a Kuhn cube on the CPU and checks that it owns its voxels exactly once.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "tetvox/config.hpp"
#include "tetvox/cpu.hpp"
#include "tetvox/device.hpp"

namespace {

// Cube [0, 3]^3 split into 6 positively oriented tets around the diagonal from vertex 0 to 7.
// Vertex i sits at 3 * (bit 0, bit 1, bit 2) of i.
constexpr int64_t num_vertices = 8;
constexpr int64_t num_tets = 6;
constexpr int32_t tetrahedra[num_tets][4] = {{0, 1, 3, 7}, {0, 5, 1, 7}, {0, 3, 2, 7},
                                             {0, 2, 6, 7}, {0, 4, 5, 7}, {0, 6, 4, 7}};

bool rasterize_kuhn_cube() {
  std::vector<float> positions;
  for (int i = 0; i < num_vertices; ++i) {
    for (int d = 0; d < 3; ++d) {
      positions.push_back(3.0F * static_cast<float>((i >> d) & 1));
    }
  }
  std::vector<int64_t> normals(num_tets * 12), offsets(num_tets * 4), box_lo(num_tets * 3),
      box_hi(num_tets * 3);
  std::vector<float> fractions(num_tets * 4), bary_scale(num_tets);
  std::vector<tetvox::TetStatus> status(num_tets);
  const tetvox::TetSetupView<float> setup{num_tets,         normals.data(),    offsets.data(),
                                          fractions.data(), bary_scale.data(), box_lo.data(),
                                          box_hi.data(),    status.data()};
  const tetvox::MeshView<float> mesh{positions.data(), &tetrahedra[0][0], num_vertices, num_tets};
  tetvox::cpu::setup_tets(mesh, tetvox::default_snap_bits, setup);
  const tetvox::SkippedTets skipped = tetvox::cpu::count_skipped(status.data(), num_tets);

  const tetvox::GridShape shape = {4, 4, 4};
  std::vector<int32_t> tet_id(64), claims(64);
  std::vector<float> bary(64 * 3);
  tetvox::cpu::rasterize(setup, shape, tet_id.data(), bary.data());
  tetvox::cpu::count_claims(setup, shape, claims.data());

  // The half-open box [0, 3)^3 is owned exactly once; the layers at index 3 are not owned.
  int64_t errors = 0;
  for (int64_t i = 0; i < 64; ++i) {
    const bool inside = i / 16 < 3 && (i / 4) % 4 < 3 && i % 4 < 3;
    errors += (claims[i] != (inside ? 1 : 0)) || ((tet_id[i] >= 0) != inside);
  }
  std::printf("tetvox: Kuhn cube, %lld inverted, %lld outside window, %lld voxel errors\n",
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
  return devices_ok && rasterize_kuhn_cube() ? 0 : 1;
}
