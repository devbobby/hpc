#include <iostream>
#include <cuda_runtime.h>
using namespace std;

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        cerr << "No CUDA-capable NVIDIA GPU detected by the CUDA runtime.\n";
        cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    for (int d = 0; d < count; ++d) {
        cudaDeviceProp p{};
        cudaGetDeviceProperties(&p, d);
        cout << "GPU " << d << ": " << p.name << "\n";
        cout << "Global memory (MiB): "
             << (p.totalGlobalMem / (1024.0 * 1024.0)) << "\n";
        cout << "Compute capability: " << p.major << "." << p.minor << "\n";
        cout << "Multiprocessors (SMs): " << p.multiProcessorCount << "\n";
        cout << "Maximum threads per block: " << p.maxThreadsPerBlock << "\n";
        cout << "Maximum grid dimensions: "
             << p.maxGridSize[0] << " x " << p.maxGridSize[1]
             << " x " << p.maxGridSize[2] << "\n";
    }
    return 0;
}
