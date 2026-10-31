#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void helloKernel() {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from GPU Thread %d (block %d, local thread %d)\n",
           globalId, blockIdx.x, threadIdx.x);
}

int main(int argc, char* argv[]) {
    int blocks = (argc > 1) ? atoi(argv[1]) : 1;
    int threads = (argc > 2) ? atoi(argv[2]) : 8;
    if (blocks < 1 || threads < 1) {
        cerr << "Usage: hello_cuda [blocks>=1] [threads_per_block>=1]\n";
        return 1;
    }
    helloKernel<<<blocks, threads>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "Kernel launch error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cerr << "Kernel execution error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }
    return 0;
}
