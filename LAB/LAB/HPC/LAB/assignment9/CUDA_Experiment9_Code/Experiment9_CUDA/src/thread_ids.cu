#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void showIds() {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Block ID = %d, Thread ID in block = %d, Global ID = %d\n",
           blockIdx.x, threadIdx.x, globalId);
}

int main() {
    int blocks = 2;
    int threadsPerBlock = 4;
    cout << "Launching " << blocks << " blocks x "
         << threadsPerBlock << " threads = "
         << blocks * threadsPerBlock << " total threads\n";
    showIds<<<blocks, threadsPerBlock>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }
    return 0;
}
