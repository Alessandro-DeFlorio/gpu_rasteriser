cmake_minimum_required(VERSION 3.26)
# Must precede enable_language(CUDA): enabling CUDA replaces an unset value with the compiler
# default (sm_75), which needs PTX JIT on newer GPUs (docs/knowledge/build.md). A parent project's
# value wins.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
# The version is the single source of truth: pyproject.toml reads it from this line.
project(tetvox VERSION 0.0.1 LANGUAGES CXX)

# ADR 0007/0008: OFF gives a CPU-only build that needs no CUDA compiler.
option(TETVOX_ENABLE_CUDA "Build the CUDA backend (needs a CUDA compiler)" ON)
if(TETVOX_ENABLE_CUDA)
  enable_language(CUDA)
endif()
# Read by config.hpp.in (#cmakedefine01) and exported to consumers by tetvoxConfig.cmake.
set(TETVOX_CUDA_ENABLED ${TETVOX_ENABLE_CUDA})

if(DEFINED SKBUILD)
  set(tetvox_python_default ON)
else()
  set(tetvox_python_default OFF)
endif()
option(TETVOX_BUILD_PYTHON "Build the tetvox._core Python module (needs Python and nanobind)"
       ${tetvox_python_default})

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Core library (ADR 0004): public headers in csrc/include, kernels in csrc/src. It leaves the choice
# of CUDA runtime (static or shared cudart) to whatever links it.
add_library(tetvox_core STATIC csrc/src/cpu.cpp)
add_library(tetvox::tetvox ALIAS tetvox_core)
# CPU backend parallelism (ADR 0007/0008). Without OpenMP the same code runs serially. A private
# dependency of a static library is exported as link-only, so consumers need no -fopenmp.
find_package(OpenMP COMPONENTS CXX)
set(TETVOX_HAS_OPENMP ${OpenMP_CXX_FOUND})
if(TETVOX_HAS_OPENMP)
  target_link_libraries(tetvox_core PRIVATE OpenMP::OpenMP_CXX)
endif()
if(TETVOX_CUDA_ENABLED)
  target_sources(tetvox_core PRIVATE csrc/src/device.cu)
  target_compile_features(tetvox_core PUBLIC cuda_std_20)
else()
  target_sources(tetvox_core PRIVATE csrc/src/device_cpu_only.cpp)
endif()
# Globbed so that a new public header can never be left out of the installed package.
file(GLOB_RECURSE tetvox_public_headers CONFIGURE_DEPENDS csrc/include/tetvox/*.hpp
     csrc/include/tetvox/*.cuh)
set(tetvox_generated_include ${CMAKE_CURRENT_BINARY_DIR}/include)
configure_file(cmake/config.hpp.in ${tetvox_generated_include}/tetvox/config.hpp @ONLY)
target_sources(
  tetvox_core
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         csrc/include
         ${tetvox_generated_include}
         FILES
         ${tetvox_public_headers}
         ${tetvox_generated_include}/tetvox/config.hpp)
target_compile_features(tetvox_core PUBLIC cxx_std_20)
set_target_properties(
  tetvox_core
  PROPERTIES EXPORT_NAME tetvox
             OUTPUT_NAME tetvox
             POSITION_INDEPENDENT_CODE ON)

install(
  TARGETS tetvox_core
  EXPORT tetvoxTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT cpp
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT cpp)
set(tetvox_cmake_dir ${CMAKE_INSTALL_LIBDIR}/cmake/tetvox)
install(
  EXPORT tetvoxTargets
  NAMESPACE tetvox::
  DESTINATION ${tetvox_cmake_dir}
  COMPONENT cpp)
configure_package_config_file(cmake/tetvoxConfig.cmake.in tetvoxConfig.cmake
                              INSTALL_DESTINATION ${tetvox_cmake_dir})
write_basic_package_version_file(tetvoxConfigVersion.cmake COMPATIBILITY SameMinorVersion)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/tetvoxConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/tetvoxConfigVersion.cmake
  DESTINATION ${tetvox_cmake_dir}
  COMPONENT cpp)

if(TETVOX_BUILD_PYTHON)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(_core NB_STATIC csrc/bindings/module.cpp)
  target_link_libraries(_core PRIVATE tetvox_core)
  # Static cudart: no clash with the CUDA runtime bundled in torch wheels (ADR 0001).
  set_target_properties(_core PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  install(
    TARGETS _core
    LIBRARY DESTINATION tetvox
    COMPONENT python)
endif()
