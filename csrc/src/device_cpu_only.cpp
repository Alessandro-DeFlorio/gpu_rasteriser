// device_count() of a CPU-only build (TETVOX_ENABLE_CUDA=OFF, ADR 0008).
#include "tetvox/device.hpp"

namespace tetvox {

int device_count() { return 0; }

}  // namespace tetvox
