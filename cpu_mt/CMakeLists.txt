cmake_minimum_required(VERSION 3.16)
project(cpu_mt_bench LANGUAGES CXX)

add_library(cpu_mt SHARED
  cpu_mt_api.cc
  cpu_mt_runner.cc
  cpu_mt_runner.h
  # AVX-512 kernels reused directly
  ${CMAKE_SOURCE_DIR}/cpu_avx512/algorithms_opt.cc
  ${CMAKE_SOURCE_DIR}/cpu_avx512/algorithms_opt.hpp
)

target_include_directories(cpu_mt PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/cpu_avx512
  ${CMAKE_SOURCE_DIR}/common_abi
  ${CMAKE_SOURCE_DIR}
)

target_compile_features(cpu_mt PRIVATE cxx_std_17)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(cpu_mt PRIVATE
    -O3 -march=native -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma
  )
endif ()

target_link_libraries(cpu_mt PRIVATE pthread)
