cmake_minimum_required(VERSION 3.24)
# Set before project() initializes the Apple toolchain.
if(CMAKE_HOST_APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum supported macOS target")
endif()
project(scimetal VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Descriptor ABI (sm.h): structural validation only, no device.
add_library(sm_contract STATIC src/runtime/buffer_contract.cpp)
target_include_directories(sm_contract PUBLIC include)
set_target_properties(sm_contract PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)

option(SM_ENABLE_METAL "Enable shared Metal storage and GPU routes in the native core" ON)
# Vendored Apple metal-cpp headers (third_party/metal-cpp/PINNED_VERSION.md).
set(SM_METAL_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/metal-cpp"
  CACHE PATH "metal-cpp header directory")

# The native core: runtime (vendored from metal-arrow, see src/runtime/PROVENANCE.md),
# router, CPU kernels and GPU routes behind the C ABI in include/.
# Built once as an object library so the shipped shared library exports only
# the public C ABI (`sm_*`) while internal C++ unit tests can link against
# `mscore_test` to exercise `sm::detail::*` directly.
add_library(mscore_obj OBJECT
  src/runtime/storage.cpp src/core/array.cpp src/core/router.cpp src/core/dense_correlate.cpp)
set_target_properties(mscore_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mscore_obj PUBLIC include)
target_include_directories(mscore_obj PRIVATE src)
target_compile_options(mscore_obj PRIVATE -Wall -Wextra -Wpedantic)
# The CPU FIR kernel reproduces an explicit sequence of rounded operations.
set_property(SOURCE src/core/dense_correlate.cpp APPEND PROPERTY COMPILE_OPTIONS
    -fno-fast-math -ffp-contract=off)
if(APPLE)
  # New LAPACK/CBLAS interface (macOS 13.3+).
  target_compile_definitions(mscore_obj PRIVATE ACCELERATE_NEW_LAPACK=1)
endif()

# Numerical families register their sources in src/<family>/sources.cmake by
# appending to these lists (paths relative to the project root):
#   SM_FAMILY_SOURCES        C++ built in every configuration (C ABI, CPU engine, routing)
#   SM_FAMILY_GPU_SOURCES    C++ built only with Metal (GPU routes)
#   SM_METAL_SOURCES         .metal kernels linked into the embedded metallib
#   SM_METAL_INCLUDE_DIRS    extra include directories for the Metal compiler
# and their tests in src/<family>/tests.cmake, included after enable_testing().
set(SM_FAMILIES "ndimage;fft;signal;linalg;sparse" CACHE STRING
  "Numerical families to build (a subset isolates one family while others are in flux)")
set(SM_FAMILY_SOURCES)
set(SM_FAMILY_GPU_SOURCES)
set(SM_METAL_SOURCES src/kernels/smoke.metal src/core/dense_correlate.metal)
set(SM_METAL_INCLUDE_DIRS src/kernels src/core)
foreach(_family IN LISTS SM_FAMILIES)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/${_family}/sources.cmake)
    include(src/${_family}/sources.cmake)
  endif()
endforeach()
target_sources(mscore_obj PRIVATE ${SM_FAMILY_SOURCES})

if(SM_ENABLE_METAL)
  if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    message(FATAL_ERROR "Shared Metal storage requires Apple Silicon macOS")
  endif()
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET OR CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS "15.0")
    message(FATAL_ERROR
      "SM_ENABLE_METAL requires CMAKE_OSX_DEPLOYMENT_TARGET >= 15.0 (got '${CMAKE_OSX_DEPLOYMENT_TARGET}')")
  endif()
  target_compile_definitions(mscore_obj PRIVATE SM_ENABLE_METAL=1 METALCPP_SYMBOL_VISIBILITY_HIDDEN=1)
  if(NOT EXISTS "${SM_METAL_CPP_DIR}/Metal/Metal.hpp")
    message(FATAL_ERROR "Set SM_METAL_CPP_DIR to a metal-cpp header checkout")
  endif()
  # Kernels -> metallib -> C array embedded in the library, loaded on the first
  # GPU route. The Metal compiler defaults to fast math; results must not
  # depend on reassociation, so disable it and floating-point contraction.
  find_program(XCRUN xcrun REQUIRED)
  # Every kernel header and included MLX shader: an edit to any recompiles every kernel.
  file(GLOB_RECURSE SM_METAL_HEADERS CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mlx-kernels/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mlx-kernels/*.metal)
  set(_sm_metal_includes)
  foreach(_dir IN LISTS SM_METAL_INCLUDE_DIRS)
    list(APPEND _sm_metal_includes -I ${CMAKE_CURRENT_SOURCE_DIR}/${_dir})
  endforeach()
  set(_sm_air_files)
  foreach(_source IN LISTS SM_METAL_SOURCES)
    string(REGEX REPLACE "\\.metal$" "" _stem "${_source}")
    string(REPLACE "/" "_" _stem "${_stem}")
    set(_air ${CMAKE_CURRENT_BINARY_DIR}/kernels/${_stem}.air)
    add_custom_command(OUTPUT ${_air}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/kernels
      COMMAND ${XCRUN} -sdk macosx metal -std=metal3.1 -O2
              -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}
              -fno-fast-math -ffp-contract=off -Wall -Wextra
              ${_sm_metal_includes}
              -c ${CMAKE_CURRENT_SOURCE_DIR}/${_source} -o ${_air}
      DEPENDS ${_source} ${SM_METAL_HEADERS}
      COMMENT "Compiling ${_source}"
      VERBATIM)
    list(APPEND _sm_air_files ${_air})
  endforeach()
  set(SM_METALLIB ${CMAKE_CURRENT_BINARY_DIR}/kernels/scimetal.metallib)
  add_custom_command(OUTPUT ${SM_METALLIB}
    COMMAND ${XCRUN} -sdk macosx metallib ${_sm_air_files} -o ${SM_METALLIB}
    DEPENDS ${_sm_air_files}
    COMMENT "Linking scimetal.metallib"
    VERBATIM)
  set(SM_METALLIB_C ${CMAKE_CURRENT_BINARY_DIR}/kernels/metallib_embedded.c)
  add_custom_command(OUTPUT ${SM_METALLIB_C}
    COMMAND ${CMAKE_COMMAND} -DIN=${SM_METALLIB} -DOUT=${SM_METALLIB_C} -DNAME=sm_metallib
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedMetallib.cmake
    DEPENDS ${SM_METALLIB} cmake/EmbedMetallib.cmake
    COMMENT "Embedding scimetal.metallib"
    VERBATIM)
  target_sources(mscore_obj PRIVATE src/runtime/metal_storage.cpp src/runtime/metal_compute.cpp
    src/core/dense_correlate_gpu.cpp ${SM_FAMILY_GPU_SOURCES} ${SM_METALLIB_C})
  # Third-party headers: their pedantic warnings are not ours.
  target_include_directories(mscore_obj SYSTEM PRIVATE "${SM_METAL_CPP_DIR}")
  target_compile_options(mscore_obj PRIVATE -fblocks)
else()
  target_sources(mscore_obj PRIVATE src/runtime/metal_disabled.cpp)
endif()

set(_sm_core_libs sm_contract Threads::Threads)
if(APPLE)
  list(APPEND _sm_core_libs "-framework Accelerate")
endif()
if(SM_ENABLE_METAL)
  list(APPEND _sm_core_libs "-framework Metal" "-framework Foundation")
endif()

add_library(smcore SHARED $<TARGET_OBJECTS:mscore_obj>)
add_library(scimetal::core ALIAS smcore)
set_target_properties(smcore PROPERTIES OUTPUT_NAME scimetal)
target_include_directories(smcore PUBLIC include)
target_link_libraries(smcore PRIVATE ${_sm_core_libs})
if(APPLE)
  # Export only the public C ABI (`sm_*`); internal C++ and metal-cpp symbols stay hidden.
  target_link_options(smcore PRIVATE "LINKER:-exported_symbol,_sm_*")
endif()

# Static archive of the same objects for native C++ unit tests that inspect internal headers.
add_library(mscore_test STATIC $<TARGET_OBJECTS:mscore_obj>)
target_include_directories(mscore_test PUBLIC include)
target_link_libraries(mscore_test PUBLIC ${_sm_core_libs})

# The wheel carries the native core next to the Python package, where
# scimetal._native looks for it.
install(TARGETS smcore LIBRARY DESTINATION scimetal COMPONENT python)

enable_testing()
# Registers one native test executable under the GPU configurations it supports:
#   PLAIN      <name>, the executable as built
#   METAL      <name>_shared_storage, with METAL_ARG (Metal builds only)
#   VALIDATED  <name>_validated, under the Metal API validator, which aborts on
#              misuse, and shader validation, whose reports fail the test
#              (Metal builds only; METAL_ARG is passed when given)
# sm_add_gpu_test(<name> <target> [PLAIN] [METAL] [VALIDATED] [METAL_ARG <arg>] [ARGS <args>...])
function(sm_add_gpu_test name target)
  cmake_parse_arguments(SM "PLAIN;METAL;VALIDATED" "METAL_ARG" "ARGS" ${ARGN})
  if(SM_PLAIN)
    add_test(NAME ${name} COMMAND ${target} ${SM_ARGS})
  endif()
  if(SM_ENABLE_METAL AND SM_METAL)
    add_test(NAME ${name}_shared_storage COMMAND ${target} ${SM_METAL_ARG} ${SM_ARGS})
  endif()
  if(SM_ENABLE_METAL AND SM_VALIDATED)
    add_test(NAME ${name}_validated COMMAND ${target} ${SM_METAL_ARG} ${SM_ARGS})
    set_tests_properties(${name}_validated PROPERTIES
      ENVIRONMENT "MTL_DEBUG_LAYER=1;MTL_SHADER_VALIDATION=1;MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1"
      FAIL_REGULAR_EXPRESSION "executing kernel function;Invalid device;Invalid threadgroup")
  endif()
endfunction()

# Runtime tests vendored with the runtime.
add_executable(sm_contract_test tests/native/contract_test.cpp)
target_link_libraries(sm_contract_test PRIVATE sm_contract)
add_test(NAME descriptor_contract COMMAND sm_contract_test)
add_executable(sm_pipeline_cache_test tests/native/pipeline_cache_test.cpp)
target_include_directories(sm_pipeline_cache_test PRIVATE src/runtime)
target_link_libraries(sm_pipeline_cache_test PRIVATE Threads::Threads)
add_test(NAME pipeline_cache COMMAND sm_pipeline_cache_test)
foreach(name IN ITEMS storage metal_storage buffer_cache)
  add_executable(sm_${name}_test tests/native/${name}_test.cpp)
  target_link_libraries(sm_${name}_test PRIVATE mscore_test Threads::Threads)
endforeach()
add_test(NAME storage COMMAND sm_storage_test)
add_test(NAME buffer_cache COMMAND sm_buffer_cache_test)
add_executable(sm_array_test tests/native/array_test.cpp)
target_include_directories(sm_array_test PRIVATE src)
target_link_libraries(sm_array_test PRIVATE mscore_test Threads::Threads)
add_test(NAME array COMMAND sm_array_test)
if(SM_ENABLE_METAL)
  add_test(NAME shared_metal_storage COMMAND sm_metal_storage_test)
  add_test(NAME buffer_cache_shared_storage COMMAND sm_buffer_cache_test --metal)
else()
  add_test(NAME disabled_metal_storage COMMAND sm_metal_storage_test --disabled)
endif()
add_executable(sm_metal_cache_policy_test tests/native/metal_cache_policy_test.cpp)
target_include_directories(sm_metal_cache_policy_test PRIVATE src/runtime)
target_link_libraries(sm_metal_cache_policy_test PRIVATE mscore_test)
if(SM_ENABLE_METAL)
  target_compile_definitions(sm_metal_cache_policy_test PRIVATE SM_TEST_METAL=1 METALCPP_SYMBOL_VISIBILITY_HIDDEN=1)
  target_include_directories(sm_metal_cache_policy_test SYSTEM PRIVATE "${SM_METAL_CPP_DIR}")
endif()
sm_add_gpu_test(metal_cache_policy sm_metal_cache_policy_test PLAIN VALIDATED)

# SciMetal's own runtime tests.
if(SM_ENABLE_METAL)
  add_executable(sm_compute_smoke_test tests/native/compute_smoke_test.cpp)
  target_include_directories(sm_compute_smoke_test PRIVATE src/runtime)
  target_include_directories(sm_compute_smoke_test SYSTEM PRIVATE "${SM_METAL_CPP_DIR}")
  target_compile_definitions(sm_compute_smoke_test PRIVATE METALCPP_SYMBOL_VISIBILITY_HIDDEN=1)
  target_link_libraries(sm_compute_smoke_test PRIVATE mscore_test Threads::Threads)
  sm_add_gpu_test(compute_smoke sm_compute_smoke_test PLAIN VALIDATED)
endif()

# Family tests.
foreach(_family IN LISTS SM_FAMILIES)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/${_family}/tests.cmake)
    include(src/${_family}/tests.cmake)
  endif()
endforeach()
