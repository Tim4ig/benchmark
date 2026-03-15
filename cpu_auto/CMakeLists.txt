cmake_minimum_required(VERSION 3.16)
project(cpu_auto_bench LANGUAGES CXX)

add_library(cpu_auto SHARED
  cpu_auto_api.cc
  cpu_auto_runner.cc
  cpu_auto_runner.h
  algos/algorithm.h
  algos/algorithms.h
  ../cpu_scalar/algos/algorithm.h
  ../cpu_scalar/algos/algorithms.h
  ../cpu_scalar/algos/algorithms.cc
  ../cpu_scalar/algos/vecadd.cc
  ../cpu_scalar/algos/reduce.cc
  ../cpu_scalar/algos/prefix.cc
  ../cpu_scalar/algos/histogram.cc
  ../cpu_scalar/algos/conv2d.cc
  ../cpu_scalar/algos/spmv.cc
  ../cpu_scalar/algos/matmul.cc
  ../cpu_scalar/algos/blackscholes.cc
  ../cpu_scalar/algos/bitonic_sort.cc
  ../cpu_scalar/algos/nbody.cc
)

target_include_directories(cpu_auto PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpu_scalar
  ${CMAKE_CURRENT_SOURCE_DIR}/../common_abi
)

target_compile_features(cpu_auto PRIVATE cxx_std_17)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Auto-vectorization: let the compiler generate optimal SIMD without manual intrinsics.
  # -march=native enables all CPU extensions (AVX-512 on supported CPUs).
  # -fopt-info-vec-optimized can be added to see what got vectorized.
  target_compile_options(cpu_auto PRIVATE
    -O3
    -march=native
    -ftree-vectorize
    -ffast-math
  )
endif ()
