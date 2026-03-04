#include "convolution.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

struct TestCase {
    // Note: while convolution doesn't require the input and kernel to be square,
    // this keeps the test cases simpler
    int N;
    int K;
    std::vector<float> input;
    std::vector<float> kernel;
};

static constexpr float COMPARE_EPSILON = 1e-2f;

static double elapsed_ms(std::chrono::steady_clock::time_point a,
                         std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static double compare(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size())
        return -1.0; // Size mismatch?
    
    double max_diff = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = std::abs(a[i] - b[i]);
        if (diff > COMPARE_EPSILON && diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

// Testing program to benchmark direct vs FFT-based convolution
// Outputs a CSV to stdout
int main() {
    Engine eng = make_engine();

    std::mt19937 rng(0); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Build test cases
    std::vector<TestCase> cases;
    for (auto pair : std::vector<std::pair<int, int>>{
             {10, 3},
             {10, 10},
             {100, 3},
             {100, 10},
             {100, 50},
             {100, 100},
             {500, 3},
             {500, 10},
             {500, 50},
             {500, 100},
             {500, 500},
             {1000, 3},
             {1000, 10},
             {1000, 50},
             {1000, 100},
             {1000, 500},
             {1000, 1000},
         }) {
        TestCase tc;
        tc.N = pair.first;
        tc.K = pair.second;
        tc.input.resize(tc.N * tc.N);
        tc.kernel.resize(tc.K * tc.K);
        for (float &v : tc.input)
            v = dist(rng);
        for (float &v : tc.kernel)
            v = dist(rng);
        cases.push_back(std::move(tc));
    }

    std::cout << "N,K,direct_ms,fft_ms,fft_direct_ratio,diff\n";

    for (const TestCase &tc : cases) {
        double direct_ms = 0.0, fft_ms = 0.0;
        std::vector<float> out_direct, out_fft;

        {
            // Warmup runs (for JIT and other effects)
            std::vector<float> warmup;
            convolve_direct(eng, tc.input, tc.kernel, warmup, tc.N, tc.N, tc.K, tc.K);
            convolve_direct(eng, tc.input, tc.kernel, warmup, tc.N, tc.N, tc.K, tc.K);
            eng.queue.finish();

            // Actual timed direct run
            auto t0 = std::chrono::steady_clock::now();
            convolve_direct(eng, tc.input, tc.kernel, out_direct, tc.N, tc.N, tc.K, tc.K);
            eng.queue.finish();
            direct_ms = elapsed_ms(t0, std::chrono::steady_clock::now());
        }

        {
            // Warmup runs (for JIT and other effects)
            std::vector<float> warmup;
            convolve_fft(eng, tc.input, tc.kernel, warmup, tc.N, tc.N, tc.K, tc.K);
            convolve_fft(eng, tc.input, tc.kernel, warmup, tc.N, tc.N, tc.K, tc.K);
            eng.queue.finish();

            // Actual timed FFT run
            auto t0 = std::chrono::steady_clock::now();
            convolve_fft(eng, tc.input, tc.kernel, out_fft, tc.N, tc.N, tc.K, tc.K);
            eng.queue.finish();
            fft_ms = elapsed_ms(t0, std::chrono::steady_clock::now());
        }

        double compare_result = compare(out_direct, out_fft);
        std::cout << std::fixed << std::setprecision(4)
                  << tc.N << "," << tc.K << ","
                  << direct_ms << "," << fft_ms << ","
                  << (direct_ms > 0.0 ? direct_ms / fft_ms : 0.0) << ","
                  << compare_result << "\n";
    }
}
