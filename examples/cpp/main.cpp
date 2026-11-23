// Minimal consumer of the installed tetvox C++ package (ADR 0004).
#include <cstdio>

#include "tetvox/device.hpp"

int main() {
  const int devices = tetvox::device_count();
  std::printf("tetvox: %d CUDA device(s) visible\n", devices);
  return devices >= 1 ? 0 : 1;
}
