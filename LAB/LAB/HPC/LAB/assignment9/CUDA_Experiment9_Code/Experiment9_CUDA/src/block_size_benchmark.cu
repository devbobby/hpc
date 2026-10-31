#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
using namespace std;

__global__ void add(const float* a, const float* b, float* c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) c[id] = a[id] + b[id];
}

int main(int argc, char* argv[]) {
    int n = (argc > 1) ? atoi(argv[1]) : 1000000;
    if (n <= 0) { cerr << "Usage: block_size_benchmark [N]\n"; return 1; }
    const int sizes[] = {64, 128, 256, 512};
    vector<float> a(n, 1.25f), b(n, 2.5f), c(n);
    float *da, *db, *dc;
    if (cudaMalloc(&da, n*sizeof(float)) != cudaSuccess ||
        cudaMalloc(&db, n*sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dc, n*sizeof(float)) != cudaSuccess) {
        cerr << "Could not allocate GPU memory. Reduce N or check GPU memory.\n";
        return 1;
    }
    cudaMemcpy(da, a.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), n*sizeof(float), cudaMemcpyHostToDevice);

    cout << "N=" << n << " (record actual measurements in report)\n";
    cout << "BlockSize,Blocks,KernelTime_ms\n";
    for (int threads : sizes) {
        int blocks = (n + threads - 1) / threads;
        // Warm-up to reduce first-launch initialization effects.
        add<<<blocks, threads>>>(da, db, dc, n);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int repeat = 0; repeat < 10; ++repeat)
            add<<<blocks, threads>>>(da, db, dc, n);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        cout << threads << "," << blocks << "," << ms / 10.0f << "\n";
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
    cudaFree(da); cudaFree(db); cudaFree(dc);
    return 0;
}
