#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include <string>
#include <vector>

// A structure containing various OpenCL objects and kernels used for convolution
struct Engine {
    cl::Context context;
    cl::CommandQueue queue;
    cl::Kernel direct_k;
    cl::Kernel real_to_complex_k;
    cl::Kernel bit_reverse_rows_k;
    cl::Kernel fft_butterfly_rows_k;
    cl::Kernel bit_reverse_cols_k;
    cl::Kernel fft_butterfly_cols_k;
    cl::Kernel complex_multiply_k;
    cl::Kernel extract_normalize_k;
};

// Initialises OpenCL and builds the kernels
Engine make_engine();

// Direct convolution via the GPU direct kernel
void convolve_direct(Engine &eng, const std::vector<float> &input, const std::vector<float> &h,
                     std::vector<float> &output, int inW, int inH, int kW, int kH);

// FFT-based convolution via the GPU FFT kernels
void convolve_fft(Engine &eng, const std::vector<float> &input,
                  const std::vector<float> &kernel_data, std::vector<float> &output, int inW,
                  int inH, int kW, int kH);
