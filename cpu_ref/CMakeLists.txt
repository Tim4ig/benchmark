cmake_minimum_required(VERSION 3.16)

project(cpu_ref_bench LANGUAGES CXX)

add_library(cpu_ref SHARED
        cpu_ref_api.cc
        cpu_ref_runner.cc
        cpu_ref_runner.h
        algos/algorithm.h
        algos/algorithms.h
        algos/algorithms.cc
)

target_include_directories(cpu_ref PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../common_abi
)

target_compile_features(cpu_ref PRIVATE cxx_std_17)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cpu_ref PRIVATE -O0 -fno-tree-vectorize -fno-unroll-loops)
elseif (MSVC)
    target_compile_options(cpu_ref PRIVATE /Od)
endif ()
