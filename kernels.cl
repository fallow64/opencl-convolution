kernel void vector_add(
    __global const float* A,
    __global const float* B,
    __global float* C
) {
    int i = get_global_id(0);
    C[i] = A[i] + B[i];
}

kernel void convolution2d_naive(
    __global const float* input,
    __global const float* kernel_data,
    __global float* output,
    const int inputWidth,
    const int inputHeight,
    const int kernelWidth,
    const int kernelHeight
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    float sum = 0.0f;
    for (int j = 0; j < kernelHeight; j++) {
        for (int i = 0; i < kernelWidth; i++) {
            int inputX = x + i - kernelWidth / 2;
            int inputY = y + j - kernelHeight / 2;

            // Check bounds
            if (inputX >= 0 && inputX < inputWidth && inputY >= 0 && inputY < inputHeight) {
                sum += input[inputY * inputWidth + inputX] * kernel_data[j * kernelWidth + i];
            }
        }
    }

    output[y * inputWidth + x] = sum;
}

kernel void convolution2d_fft(
    __global const float* input,
    __global const float* kernel_data,
    __global float* output,
    const int inputWidth,
    const int inputHeight,
    const int kernelWidth,
    const int kernelHeight
) {
    // Placeholder for FFT-based convolution implementation
    // This is a complex algorithm and would require additional helper functions for FFT and IFFT
}
