#include <nanobind/nanobind.h>

#include "tetvox/config.hpp"
#include "tetvox/device.hpp"

namespace nb = nanobind;

NB_MODULE(_core, m) {
  m.doc() = "tetvox C++/CUDA core";
  m.attr("cuda_enabled") = tetvox::cuda_enabled;
  m.def("device_count", &tetvox::device_count,
        "Number of visible CUDA devices (always 0 in a CPU-only build).");
}
