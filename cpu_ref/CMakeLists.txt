cmake_minimum_required(VERSION 3.16)

project(cpu_ref_bench LANGUAGES CXX)

add_library(cpu_ref SHARED
  cpu_ref_api.cc
  cpu_ref_runner.cc
  cpu_ref_runner.h
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

target_include_directories(cpu_ref PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpu_scalar
  ${CMAKE_CURRENT_SOURCE_DIR}/../common_abi
)

target_compile_features(cpu_ref PRIVATE cxx_std_17)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(cpu_ref PRIVATE -O0 -fno-tree-vectorize -fno-unroll-loops)
elseif (MSVC)
  target_compile_options(cpu_ref PRIVATE /Od)
endif ()
