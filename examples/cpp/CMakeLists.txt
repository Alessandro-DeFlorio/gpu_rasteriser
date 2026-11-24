# Packaging test for ADR 0004: a stand-alone project that consumes the installed tetvox package.
#   cmake -S examples/cpp -B <dir> -DCMAKE_PREFIX_PATH=<tetvox install prefix>
cmake_minimum_required(VERSION 3.26)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(tetvox_example LANGUAGES CXX)

# A tetvox built with CUDA ships CUDA code: linking it needs the CUDA runtime, and this project picks
# which one (CMAKE_CUDA_RUNTIME_LIBRARY, static by default). A CPU-only tetvox needs no CUDA. The
# QUIET call only reads tetvox_CUDA_ENABLED; the second one fails loudly if anything is missing.
find_package(tetvox 0.0 QUIET CONFIG)
if(tetvox_CUDA_ENABLED)
  enable_language(CUDA)
endif()
find_package(tetvox 0.0 REQUIRED CONFIG)

add_executable(tetvox_example main.cpp)
target_link_libraries(tetvox_example PRIVATE tetvox::tetvox)
