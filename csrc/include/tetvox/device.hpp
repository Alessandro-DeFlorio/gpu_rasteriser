#pragma once

namespace tetvox {

// Number of CUDA devices visible to this process (0 if no driver/device, and always 0 in a
// CPU-only build, see tetvox/config.hpp).
int device_count();

}  // namespace tetvox
