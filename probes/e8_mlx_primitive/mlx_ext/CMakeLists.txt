# E8 probe: out-of-tree MLX extension built against the installed mlx wheel.
cmake_minimum_required(VERSION 3.27)

project(e8_mlx_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ----------------------------- Dependencies -----------------------------
find_package(
  Python 3.10
  COMPONENTS Interpreter Development.Module
  REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

if(NOT MLX_BUILD_METAL)
  message(FATAL_ERROR "E8 needs an MLX wheel built with the Metal backend")
endif()

# ----------------------------- Main module -----------------------------
# The kernel metallib is NOT built here: it is compiled offline by build.sh
# (it is SciMetal's own artifact, shared with the standalone runner).
nanobind_add_module(
  _e8_mlx
  NB_STATIC
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp
  ${CMAKE_CURRENT_LIST_DIR}/e8_primitive.cpp
  ${CMAKE_CURRENT_LIST_DIR}/metal_counters.cpp)
target_link_libraries(_e8_mlx PRIVATE mlx)
# Floating-point contraction of the CPU path (eval_cpu).  The Metal compiler
# fuses `acc += w * x` into an fma even with -fno-fast-math, so "on" (clang
# fuses too) makes eval_cpu bit-identical to the GPU kernel; with "off" the
# two differ in the last ulp (max abs diff 1.9e-6 on the probe's test data).
set(E8_CPU_FP_CONTRACT "on" CACHE STRING "-ffp-contract for eval_cpu")
set_source_files_properties(
  ${CMAKE_CURRENT_LIST_DIR}/e8_primitive.cpp
  PROPERTIES COMPILE_OPTIONS "-ffp-contract=${E8_CPU_FP_CONTRACT}")

# ------------------- Documented nanobind sharing path -------------------
nanobind_add_module(
  _e8_nbnative
  NB_STATIC
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/native_sharing.cpp)
target_link_libraries(_e8_nbnative PRIVATE mlx)

# libmlx.dylib has install name @rpath/libmlx.dylib; point at the wheel.
foreach(tgt _e8_mlx _e8_nbnative)
  set_target_properties(${tgt} PROPERTIES BUILD_RPATH "${MLX_LIBRARY_DIRS}")
endforeach()
