cmake_minimum_required(VERSION 3.16)

project(cpugpu_bench LANGUAGES CXX)

add_subdirectory(cpu_ref)
add_subdirectory(cpu_avx512)
add_subdirectory(cpu_auto)
add_subdirectory(vkbench)
add_subdirectory(cpu_mt)
add_subdirectory(hybrid)
add_subdirectory(orchestrator)
