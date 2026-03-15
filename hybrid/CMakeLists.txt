cmake_minimum_required(VERSION 3.16)
project(hybrid_bench LANGUAGES CXX)

find_package(Vulkan REQUIRED)

add_library(hybrid SHARED
  src/hybrid_api.cc
  src/hybrid_runner.cc
  src/hybrid_runner.h
  ${CMAKE_SOURCE_DIR}/vkbench/src/vk_context.cc
  ${CMAKE_SOURCE_DIR}/vkbench/src/vk_buffer.cc
  ${CMAKE_SOURCE_DIR}/vkbench/src/vk_pipeline.cc
)

add_dependencies(hybrid vkbench_shaders)

target_include_directories(hybrid PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/common_abi
  ${CMAKE_SOURCE_DIR}/vkbench/src
  ${CMAKE_SOURCE_DIR}
)

target_compile_features(hybrid PRIVATE cxx_std_17)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(hybrid PRIVATE
    -O3
    -march=native
    -mavx512f
    -mavx512dq
    -mavx512bw
    -mavx512vl
    -mfma
  )
endif ()

target_link_libraries(hybrid PRIVATE Vulkan::Vulkan)

if (UNIX AND NOT APPLE)
  target_link_libraries(hybrid PRIVATE dl pthread)
endif ()
