#include <nanobind/nanobind.h>

#include "tetvox/device.hpp"

namespace nb = nanobind;

NB_MODULE(_core, m) {
  m.doc() = "tetvox C++/CUDA core";
  m.def("device_count", &tetvox::device_count, "Number of visible CUDA devices.");
}
