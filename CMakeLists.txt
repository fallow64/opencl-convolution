cmake_minimum_required(VERSION 3.16)
project(opencl_convolution CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCL REQUIRED)

add_executable(convolution_tests
    src/test.cpp
    src/convolution.cpp
)

target_include_directories(convolution_tests PRIVATE include)
target_compile_options(convolution_tests PRIVATE -O2 -Wall -Wextra)
target_link_libraries(convolution_tests PRIVATE OpenCL::OpenCL)
