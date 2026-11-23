cmake_minimum_required(VERSION 3.26)
project(tetvox LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Static cudart: no clash with the CUDA runtime bundled in torch wheels.
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Core library: header-only math in csrc/include, kernels in csrc/src.
add_library(tetvox_core STATIC csrc/src/device.cu)
target_include_directories(tetvox_core PUBLIC csrc/include)
set_target_properties(tetvox_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

nanobind_add_module(_core NB_STATIC csrc/bindings/module.cpp)
target_link_libraries(_core PRIVATE tetvox_core)

install(TARGETS _core LIBRARY DESTINATION tetvox)
