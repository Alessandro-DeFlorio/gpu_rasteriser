#include <cuda_runtime.h>

#include "tetvox/device.hpp"

namespace tetvox {

int device_count() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    return 0;
  }
  return count;
}

}  // namespace tetvox
