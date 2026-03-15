cmake_minimum_required(VERSION 3.16)

project(bench_orchestrator LANGUAGES CXX)

add_executable(bench_orchestrator
  orchestrator.cc
)

target_compile_features(bench_orchestrator PRIVATE cxx_std_17)

target_include_directories(bench_orchestrator PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../common_abi
)

if (UNIX AND NOT APPLE)
  target_link_libraries(bench_orchestrator PRIVATE dl)
endif ()
