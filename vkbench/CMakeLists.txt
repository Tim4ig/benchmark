cmake_minimum_required(VERSION 3.16)

project(vkbench LANGUAGES CXX)

find_package(Vulkan REQUIRED)
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

set(SHADER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BIN_DIR})

set(SHADERS
        vecadd.comp
        reduce.comp
        scan_block.comp
        scan_add.comp
        hist.comp
        conv2d.comp
        spmv.comp
        matmul.comp
)

set(SPIRV_FILES)
foreach (SHADER ${SHADERS})
    get_filename_component(SHADER_BASE ${SHADER} NAME_WE)
    set(SPV ${SHADER_BIN_DIR}/${SHADER_BASE}.spv)
    add_custom_command(
            OUTPUT ${SPV}
            COMMAND ${GLSLANG_VALIDATOR} -V ${SHADER_SRC_DIR}/${SHADER} -o ${SPV}
            DEPENDS ${SHADER_SRC_DIR}/${SHADER}
            COMMENT "Compiling ${SHADER}"
    )
    list(APPEND SPIRV_FILES ${SPV})
endforeach ()

add_custom_target(vkbench_shaders DEPENDS ${SPIRV_FILES})

add_library(vkbench SHARED
        src/main.cc
        src/vk_context.cc
        src/vk_buffer.cc
        src/vk_pipeline.cc
)

add_dependencies(vkbench vkbench_shaders)

target_compile_features(vkbench PRIVATE cxx_std_17)

target_include_directories(vkbench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../common_abi
)

target_link_libraries(vkbench PRIVATE Vulkan::Vulkan)

if(UNIX AND NOT APPLE)
  target_link_libraries(vkbench PRIVATE dl)
endif()
