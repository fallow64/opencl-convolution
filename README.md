# opencl-convolution

A comparison of direct vs FFT-based convolution using OpenCL.

## Building

This program requires the OpenCL headers and libraries, a C++ compiler, and CMake.

To build:
```sh
cmake -B build
cmake --build build
```

## Running

The `test` executable runs a series of convolution benchmarks with different inputs and kernels. It will print out a CSV report to stdout.

Additionally, the file `kernels.cl` is expected to be in your current working directory when running `test`, as it contains the OpenCL kernels used for the convolution operations.