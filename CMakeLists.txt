cmake_minimum_required(VERSION 3.16)
project(opencl_convolution CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCL REQUIRED)

# Embed the OpenCL kernels into a header file using xxd
# Could also compile the kernel to SPIR-V and embed that to save space and reduce load time
find_program(XXD_COMMAND xxd REQUIRED)
set(KERNELS_EMBED_HEADER "${CMAKE_BINARY_DIR}/kernels_embed.h")
add_custom_command(
    OUTPUT "${KERNELS_EMBED_HEADER}"
    COMMAND ${XXD_COMMAND} -i kernels.cl "${KERNELS_EMBED_HEADER}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    DEPENDS "${CMAKE_SOURCE_DIR}/kernels.cl"
    COMMENT "Embedding kernels.cl"
    VERBATIM
)

add_executable(convolution_tests
    src/test.cpp
    src/convolution.cpp
    "${KERNELS_EMBED_HEADER}"
)

target_include_directories(convolution_tests PRIVATE include "${CMAKE_BINARY_DIR}")
target_compile_options(convolution_tests PRIVATE -O2 -Wall -Wextra)
target_link_libraries(convolution_tests PRIVATE OpenCL::OpenCL)
