#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
             << " - " << cudaGetErrorString(e) << "\n"; \
        return 1; \
    } \
} while (0)

__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) C[id] = A[id] + B[id];
}

int main(int argc, char* argv[]) {
    int n = (argc > 1) ? atoi(argv[1]) : 1000000;
    int threads = (argc > 2) ? atoi(argv[2]) : 256;
    if (n <= 0 || threads <= 0) {
        cerr << "Usage: vector_add_cuda [N] [threads_per_block]\n";
        return 1;
    }

    vector<float> A(n), B(n), C(n);
    for (int i = 0; i < n; ++i) {
        A[i] = static_cast<float>(i % 100) * 0.5f;
        B[i] = static_cast<float>(i % 70) * 0.25f;
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, n * sizeof(float)));

    // Time the full host-to-device transfer, kernel, and device-to-host transfer.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int blocks = (n + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(C.data(), dC, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float totalMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));

    bool correct = true;
    for (int i = 0; i < n; ++i) {
        float expected = A[i] + B[i];
        if (fabs(C[i] - expected) > 1e-5f) { correct = false; break; }
    }

    cout << "Vector size N: " << n << "\n";
    cout << "Threads per block: " << threads << "\n";
    cout << "Blocks required: " << blocks << "\n";
    cout << "First result C[0]: " << C[0] << "\n";
    cout << "Last result C[N-1]: " << C[n - 1] << "\n";
    cout << "Correctness: " << (correct ? "PASS" : "FAIL") << "\n";
    cout << "CUDA transfer + kernel elapsed time (ms): " << totalMs << "\n";

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return correct ? 0 : 2;
}
