#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void add(const int* a, const int* b, int* c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) c[id] = a[id] + b[id];
}

int main() {
    const int N = 10;
    int hA[N], hB[N], hC[N];
    for (int i = 0; i < N; ++i) { hA[i] = i; hB[i] = 10 + i; }

    int *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaError_t e;
    if ((e = cudaMalloc(&dA, N * sizeof(int))) != cudaSuccess ||
        (e = cudaMalloc(&dB, N * sizeof(int))) != cudaSuccess ||
        (e = cudaMalloc(&dC, N * sizeof(int))) != cudaSuccess) {
        cerr << "cudaMalloc failed: " << cudaGetErrorString(e) << "\n";
        cudaFree(dA); cudaFree(dB); cudaFree(dC); return 1;
    }

    // Copy input arrays from host (CPU RAM) to device (GPU memory).
    cudaMemcpy(dA, hA, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (N + threads - 1) / threads;
    add<<<blocks, threads>>>(dA, dB, dC, N);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        cerr << "Kernel failed: " << cudaGetErrorString(e) << "\n";
        cudaFree(dA); cudaFree(dB); cudaFree(dC); return 1;
    }

    // Copy output from GPU memory back to CPU RAM.
    cudaMemcpy(hC, dC, N * sizeof(int), cudaMemcpyDeviceToHost);
    cout << "A + B = ";
    for (int i = 0; i < N; ++i) cout << hC[i] << (i + 1 == N ? "\n" : " ");

    // Release all GPU allocations when finished.
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
