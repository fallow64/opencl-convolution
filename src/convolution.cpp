#include "convolution.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>

// Reads the entire contents of a file into a string
static std::string read_file(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Failed to open: " << path << std::endl;
        exit(1);
    }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// Returns the next power of 2 greater than or equal to n
static int nextPow2(int n) {
    int p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

// Returns the integer log base 2 of n, assuming n is a power of 2
static int ilog2(int n) {
    int l = 0;
    while (n > 1) {
        l++;
        n >>= 1;
    }
    return l;
}

Engine make_engine(const std::string &kernel_path) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device device = devices[0];

    std::string deviceName;
    device.getInfo(CL_DEVICE_NAME, &deviceName);
    // Use stderr here since stdout is used for some test outputs
    std::cerr << "Using device: " << deviceName << std::endl;

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    cl::Program program(context, read_file(kernel_path));
    if (program.build(device) != CL_SUCCESS) {
        std::cerr << "Build error:" << std::endl
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }

    Engine eng;
    eng.context = context;
    eng.queue = queue;
    eng.direct_k = cl::Kernel(program, "convolution2d_direct");
    eng.real_to_complex_k = cl::Kernel(program, "real_to_complex");
    eng.bit_reverse_rows_k = cl::Kernel(program, "bit_reverse_rows");
    eng.fft_butterfly_rows_k = cl::Kernel(program, "fft_butterfly_rows");
    eng.bit_reverse_cols_k = cl::Kernel(program, "bit_reverse_cols");
    eng.fft_butterfly_cols_k = cl::Kernel(program, "fft_butterfly_cols");
    eng.complex_multiply_k = cl::Kernel(program, "complex_multiply");
    eng.extract_normalize_k = cl::Kernel(program, "extract_and_normalize");
    return eng;
}

void convolve_direct(Engine &eng, const std::vector<float> &input, const std::vector<float> &h,
                     std::vector<float> &output, int width, int height, int kW, int kH) {
    cl::Buffer inputBuf(eng.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * input.size(), const_cast<float *>(input.data()));
    cl::Buffer hBuf(eng.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * h.size(),
                    const_cast<float *>(h.data()));
    cl::Buffer outBuf(eng.context, CL_MEM_WRITE_ONLY, sizeof(float) * width * height);

    eng.direct_k.setArg(0, inputBuf);
    eng.direct_k.setArg(1, hBuf);
    eng.direct_k.setArg(2, outBuf);
    eng.direct_k.setArg(3, width);
    eng.direct_k.setArg(4, height);
    eng.direct_k.setArg(5, kW);
    eng.direct_k.setArg(6, kH);
    eng.queue.enqueueNDRangeKernel(eng.direct_k, cl::NullRange, cl::NDRange(width, height));

    output.resize(width * height);
    eng.queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, sizeof(float) * output.size(), output.data());
}

// Performs an in-place 2D FFT on a (padW x padH) complex buffer.
// Stages are serialized via sequential kernel enqueues.
static void fft2d(Engine &eng, cl::Buffer &data, int padW, int padH, bool inverse) {
    int logW = ilog2(padW);
    int logH = ilog2(padH);
    int inv = inverse ? 1 : 0;

    // Row FFT
    eng.bit_reverse_rows_k.setArg(0, data);
    eng.bit_reverse_rows_k.setArg(1, padW);
    eng.bit_reverse_rows_k.setArg(2, logW);
    eng.bit_reverse_rows_k.setArg(3, padW);
    eng.queue.enqueueNDRangeKernel(eng.bit_reverse_rows_k, cl::NullRange, cl::NDRange(padW, padH));

    for (int s = 1; s <= logW; s++) {
        eng.fft_butterfly_rows_k.setArg(0, data);
        eng.fft_butterfly_rows_k.setArg(1, s);
        eng.fft_butterfly_rows_k.setArg(2, inv);
        eng.fft_butterfly_rows_k.setArg(3, padW);
        eng.queue.enqueueNDRangeKernel(eng.fft_butterfly_rows_k, cl::NullRange,
                                       cl::NDRange(padW / 2, padH));
    }

    // Column FFT
    eng.bit_reverse_cols_k.setArg(0, data);
    eng.bit_reverse_cols_k.setArg(1, padH);
    eng.bit_reverse_cols_k.setArg(2, logH);
    eng.bit_reverse_cols_k.setArg(3, padW);
    eng.queue.enqueueNDRangeKernel(eng.bit_reverse_cols_k, cl::NullRange, cl::NDRange(padW, padH));

    for (int s = 1; s <= logH; s++) {
        eng.fft_butterfly_cols_k.setArg(0, data);
        eng.fft_butterfly_cols_k.setArg(1, s);
        eng.fft_butterfly_cols_k.setArg(2, inv);
        eng.fft_butterfly_cols_k.setArg(3, padW);
        eng.queue.enqueueNDRangeKernel(eng.fft_butterfly_cols_k, cl::NullRange,
                                       cl::NDRange(padW, padH / 2));
    }
}

void convolve_fft(Engine &eng, const std::vector<float> &input,
                  const std::vector<float> &kernel_data, std::vector<float> &output, int inW,
                  int inH, int kW, int kH) {
    // Determine padded size
    int padW = nextPow2(inW + kW - 1);
    int padH = nextPow2(inH + kH - 1);
    int padN = padW * padH;

    cl::Buffer inputBuf(eng.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * input.size(), const_cast<float *>(input.data()));
    cl::Buffer kernelBuf(eng.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         sizeof(float) * kernel_data.size(),
                         const_cast<float *>(kernel_data.data()));

    cl::Buffer inputC(eng.context, CL_MEM_READ_WRITE, sizeof(cl_float2) * padN);
    cl::Buffer kernelC(eng.context, CL_MEM_READ_WRITE, sizeof(cl_float2) * padN);
    cl::Buffer resultC(eng.context, CL_MEM_READ_WRITE, sizeof(cl_float2) * padN);

    auto fill_complex = [&](cl::Buffer &src, cl::Buffer &dst, int srcW, int srcH) {
        // Converts a real (srcW x srcH) buffer to complex (padW x padH),
        // also zero-padding as needed.
        eng.real_to_complex_k.setArg(0, src);
        eng.real_to_complex_k.setArg(1, dst);
        eng.real_to_complex_k.setArg(2, srcW);
        eng.real_to_complex_k.setArg(3, srcH);
        eng.real_to_complex_k.setArg(4, padW);
        eng.queue.enqueueNDRangeKernel(eng.real_to_complex_k, cl::NullRange,
                                       cl::NDRange(padW, padH));
    };

    fill_complex(inputBuf, inputC, inW, inH);
    fill_complex(kernelBuf, kernelC, kW, kH);

    // Perform fft2d on both input and kernel
    fft2d(eng, inputC, padW, padH, false);
    fft2d(eng, kernelC, padW, padH, false);

    // Multiply in the frequency domain
    // C[i] = A[i] * B[i]
    eng.complex_multiply_k.setArg(0, inputC);
    eng.complex_multiply_k.setArg(1, kernelC);
    eng.complex_multiply_k.setArg(2, resultC);
    eng.complex_multiply_k.setArg(3, padN);
    eng.queue.enqueueNDRangeKernel(eng.complex_multiply_k, cl::NullRange, cl::NDRange(padN));

    // Perform inverse fft2d on the result
    fft2d(eng, resultC, padW, padH, true);

    // Extract the valid region (excl. padding) and normalize by 1/(padW*padH)
    // Also complex->real
    cl::Buffer outputBuf(eng.context, CL_MEM_WRITE_ONLY, sizeof(float) * inW * inH);
    float scale = 1.0f / (float)(padW * padH);
    eng.extract_normalize_k.setArg(0, resultC);
    eng.extract_normalize_k.setArg(1, outputBuf);
    eng.extract_normalize_k.setArg(2, padW);
    eng.extract_normalize_k.setArg(3, inW);
    eng.extract_normalize_k.setArg(4, inH);
    eng.extract_normalize_k.setArg(5, (kW - 1) / 2);
    eng.extract_normalize_k.setArg(6, (kH - 1) / 2);
    eng.extract_normalize_k.setArg(7, scale);
    eng.queue.enqueueNDRangeKernel(eng.extract_normalize_k, cl::NullRange, cl::NDRange(inW, inH));

    // Read back the result (and block until ready)
    output.resize(inW * inH);
    eng.queue.enqueueReadBuffer(outputBuf, CL_TRUE, 0, sizeof(float) * output.size(),
                                output.data());
}
