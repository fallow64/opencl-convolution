kernel void convolution2d_direct(__global const float *input, __global const float *h,
                                 __global float *output, const int inputWidth,
                                 const int inputHeight, const int kernelWidth,
                                 const int kernelHeight) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    float sum = 0.0f;
    for (int j = 0; j < kernelHeight; j++) {
        for (int i = 0; i < kernelWidth; i++) {
            int inputX = x + i - kernelWidth / 2;
            int inputY = y + j - kernelHeight / 2;

            // Check bounds
            if (inputX >= 0 && inputX < inputWidth && inputY >= 0 && inputY < inputHeight) {
                sum += input[inputY * inputWidth + inputX] *
                       h[(kernelHeight - 1 - j) * kernelWidth + (kernelWidth - 1 - i)];
            }
        }
    }

    output[y * inputWidth + x] = sum;
}

// The host orchestrates the convolution as follows:
//   1. real_to_complex         - copy real input into zero-padded complex buffer
//   2. bit_reverse_rows/cols   - bit-reversal permutation (Cooley-Tukey DIT)
//   3. fft_butterfly_rows/cols - one butterfly stage, called log2(N) times
//   4. complex_multiply        - pointwise complex multiply in frequency domain
//   5. extract_and_normalize   - IFFT output -> divide by padW*padH, write real part

// Copy a real src (srcWidth x srcHeight) into a complex (float2) dst buffer
// (dstWidth x dstHeight), zero-padding the extra region.
// Launch with global size (dstWidth, dstHeight).
kernel void real_to_complex(__global const float *src, __global float2 *dst, const int srcWidth,
                            const int srcHeight, const int dstWidth) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    float val = (x < srcWidth && y < srcHeight) ? src[y * srcWidth + x] : 0.0f;
    dst[y * dstWidth + x] = (float2)(val, 0.0f);
}

// Bit-reversal permutation along rows.
// Launch with global size (N, numRows).
kernel void bit_reverse_rows(__global float2 *data, const int N, const int logN, const int W) {
    int x = get_global_id(0);
    int row = get_global_id(1);

    int rev = 0, tmp = x;
    for (int b = 0; b < logN; b++) {
        rev = (rev << 1) | (tmp & 1);
        tmp >>= 1;
    }

    if (x < rev) {
        int i = row * W + x;
        int j = row * W + rev;
        float2 t = data[i];
        data[i] = data[j];
        data[j] = t;
    }
}

// Bit-reversal permutation along columns.
// Launch with global size (numCols, N).
kernel void bit_reverse_cols(__global float2 *data, const int N, const int logN, const int W) {
    int col = get_global_id(0);
    int y = get_global_id(1);

    int rev = 0, tmp = y;
    for (int b = 0; b < logN; b++) {
        rev = (rev << 1) | (tmp & 1);
        tmp >>= 1;
    }

    if (y < rev) {
        int i = y * W + col;
        int j = rev * W + col;
        float2 t = data[i];
        data[i] = data[j];
        data[j] = t;
    }
}

// One butterfly stage of the Cooley-Tukey DIT FFT along rows.
// Call with stage = 1, 2, ..., logN.
// Launch with global size (N/2, numRows).
kernel void fft_butterfly_rows(__global float2 *data, const int stage, const int inverse,
                               const int W) {
    int i = get_global_id(0); // butterfly index within row (0 .. N/2-1)
    int row = get_global_id(1);

    int m = 1 << (stage - 1);
    int group = i / m;
    int pos = i % m;

    int idx1 = row * W + group * (m * 2) + pos;
    int idx2 = idx1 + m;

    float angle = (inverse ? 2.0f : -2.0f) * M_PI_F * (float)pos / (float)(m * 2);
    float2 twiddle = (float2)(cos(angle), sin(angle));

    float2 u = data[idx1];
    float2 d = data[idx2];
    float2 v = (float2)(d.x * twiddle.x - d.y * twiddle.y, d.x * twiddle.y + d.y * twiddle.x);

    data[idx1] = u + v;
    data[idx2] = u - v;
}

// One butterfly stage of the Cooley-Tukey DIT FFT along columns.
// Call with stage = 1, 2, ..., logN.
// Launch with global size (numCols, N/2).
kernel void fft_butterfly_cols(__global float2 *data, const int stage, const int inverse,
                               const int W) {
    int col = get_global_id(0);
    int i = get_global_id(1); // butterfly index within column (0 .. N/2-1)

    int m = 1 << (stage - 1);
    int group = i / m;
    int pos = i % m;

    int row1 = group * (m * 2) + pos;
    int row2 = row1 + m;

    float angle = (inverse ? 2.0f : -2.0f) * M_PI_F * (float)pos / (float)(m * 2);
    float2 twiddle = (float2)(cos(angle), sin(angle));

    float2 u = data[row1 * W + col];
    float2 d = data[row2 * W + col];
    float2 v = (float2)(d.x * twiddle.x - d.y * twiddle.y, d.x * twiddle.y + d.y * twiddle.x);

    data[row1 * W + col] = u + v;
    data[row2 * W + col] = u - v;
}

// Pointwise complex multiply: C[i] = A[i] * B[i].
// Launch with global size (padW * padH).
kernel void complex_multiply(__global const float2 *A, __global const float2 *B, __global float2 *C,
                             const int N) {
    int i = get_global_id(0);
    if (i >= N)
        return;
    float2 a = A[i], b = B[i];
    C[i] = (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Extract real part of IFFT output into the output image.
// Divides by (padW * padH) and offsets into the full convolution result
// to produce a "same"-size output matching the input dimensions.
// Launch with global size (dstW, dstH).
kernel void extract_and_normalize(__global const float2 *src, __global float *dst, const int srcW,
                                  const int dstW, const int dstH, const int offsetX,
                                  const int offsetY, const float scale) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= dstW || y >= dstH)
        return;
    dst[y * dstW + x] = src[(y + offsetY) * srcW + (x + offsetX)].x * scale;
}
