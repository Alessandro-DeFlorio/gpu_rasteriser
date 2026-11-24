// Minimal consumer of the installed tetvox C++ package (ADR 0004).
#include <cstdio>

#include "tetvox/config.hpp"
#include "tetvox/device.hpp"

int main() {
  const int devices = tetvox::device_count();
  std::printf("tetvox (CUDA %s): %d CUDA device(s) visible\n",
              tetvox::cuda_enabled ? "enabled" : "disabled", devices);
  // A CUDA build is checked on a machine with a GPU; a CPU-only build must report none.
  return (tetvox::cuda_enabled ? devices >= 1 : devices == 0) ? 0 : 1;
}
