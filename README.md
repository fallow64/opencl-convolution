# opencl-convolution

A comparison of direct vs FFT-based convolution using OpenCL.

## Building

This program requires OpenCL headers and libraries, a C++ compiler, and CMake.

To build:
```sh
cmake -B build
cmake --build build
```

## Running

The `test` executable runs a series of convolution benchmarks with different inputs and kernels. It will then print out a CSV report to stdout.

The report used in the slides is included in `report.csv` (ran on an Apple Macbook Pro M4).
