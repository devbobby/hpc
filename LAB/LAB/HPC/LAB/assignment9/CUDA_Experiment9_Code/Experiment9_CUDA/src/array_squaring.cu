#include <iostream>
#include <vector>
#include <cuda_runtime.h>
using namespace std;

__global__ void squareArray(const int* input, int* output, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) output[id] = input[id] * input[id];
}

int main() {
    const int N = 12;
    vector<int> input(N), cpu(N), gpu(N);
    for (int i = 0; i < N; ++i) input[i] = i + 1;

    for (int i = 0; i < N; ++i) cpu[i] = input[i] * input[i];

    int *dIn = nullptr, *dOut = nullptr;
    if (cudaMalloc(&dIn, N*sizeof(int)) != cudaSuccess ||
        cudaMalloc(&dOut, N*sizeof(int)) != cudaSuccess) {
        cerr << "GPU memory allocation failed.\n";
        cudaFree(dIn); cudaFree(dOut); return 1;
    }
    cudaMemcpy(dIn, input.data(), N*sizeof(int), cudaMemcpyHostToDevice);
    int threads = 128, blocks = (N + threads - 1) / threads;
    squareArray<<<blocks, threads>>>(dIn, dOut, N);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        cerr << "Kernel failed: " << cudaGetErrorString(e) << "\n";
        cudaFree(dIn); cudaFree(dOut); return 1;
    }
    cudaMemcpy(gpu.data(), dOut, N*sizeof(int), cudaMemcpyDeviceToHost);

    bool ok = true;
    cout << "Input: ";
    for (int x : input) cout << x << " ";
    cout << "\nCPU squares: ";
    for (int x : cpu) cout << x << " ";
    cout << "\nGPU squares: ";
    for (int x : gpu) cout << x << " ";
    for (int i = 0; i < N; ++i) if (cpu[i] != gpu[i]) ok = false;
    cout << "\nCorrectness: " << (ok ? "PASS" : "FAIL") << "\n";
    cudaFree(dIn); cudaFree(dOut);
    return ok ? 0 : 2;
}
