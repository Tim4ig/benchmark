cmake_minimum_required(VERSION 3.16)

project(cpu_avx512_bench LANGUAGES CXX)

add_library(cpu_avx512 SHARED
  algorithms_opt.cc
  algorithms_opt.hpp
  cpu_avx512_api.cc
  cpu_avx512_runner.cc
  cpu_avx512_runner.h
  algos/algorithm.h
  algos/algorithms.h
  algos/algorithms.cc
  algos/vecadd.cc
  algos/reduce.cc
  algos/prefix.cc
  algos/histogram.cc
  algos/conv2d.cc
  algos/spmv.cc
  algos/matmul.cc
  algos/blackscholes.cc
  algos/bitonic_sort.cc
  algos/nbody.cc
)

target_include_directories(cpu_avx512 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../common_abi
)

target_compile_features(cpu_avx512 PRIVATE cxx_std_17)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(cpu_avx512 PRIVATE -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma)
elseif (MSVC)
  target_compile_options(cpu_avx512 PRIVATE /O2)
endif ()
