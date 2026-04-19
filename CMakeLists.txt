cmake_minimum_required(VERSION 3.16)

project(cpugpu_bench LANGUAGES CXX)

# Build metadata embedded in orchestrator
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE BUILD_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT BUILD_GIT_HASH OR BUILD_GIT_HASH STREQUAL "")
    set(BUILD_GIT_HASH "unknown")
endif()
string(TIMESTAMP BUILD_TIMESTAMP "%Y%m%d_%H%M%S" UTC)
configure_file(
    ${CMAKE_SOURCE_DIR}/common_abi/build_info.h.in
    ${CMAKE_BINARY_DIR}/build_info.h
    @ONLY
)

add_subdirectory(cpu_ref)
add_subdirectory(cpu_avx512)
add_subdirectory(cpu_auto)
add_subdirectory(vkbench)
add_subdirectory(cpu_mt)
add_subdirectory(hybrid)
add_subdirectory(orchestrator)
