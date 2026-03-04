# opencl-convolution

A comparison of direct vs FFT-based convolution using OpenCL.

## Building

Requires the OpenCL headers and libraries, and a C++11 compiler.

To build, run `make` in the project root. This will produce an executable `test` in the `build/` directory.

## Running

The `test` executable runs a series of convolution benchmarks with different inputs and kernels. It will print out a CSV report to stdout.

Additionally, the file `kernels.cl` is expected to be in your current working directory when running `test`, as it contains the OpenCL kernels used for the convolution operations.