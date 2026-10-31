#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>
using namespace std;
using clock_type = chrono::high_resolution_clock;

__global__ void addGPU(const float* a, const float* b, float* c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) c[id] = a[id] + b[id];
}

int main(int argc, char* argv[]) {
    int n = (argc > 1) ? atoi(argv[1]) : 1000000;
    if (n <= 0) { cerr << "Usage: compare_cpu_omp_cuda [N]\n"; return 1; }
    vector<float> a(n), b(n), seq(n), par(n), gpu(n);
    for (int i = 0; i < n; ++i) {
        a[i] = float(i % 100) * 0.5f;
        b[i] = float(i % 70) * 0.25f;
    }

    auto t1 = clock_type::now();
    for (int i = 0; i < n; ++i) seq[i] = a[i] + b[i];
    auto t2 = clock_type::now();
    double seqMs = chrono::duration<double, milli>(t2 - t1).count();

    t1 = clock_type::now();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) par[i] = a[i] + b[i];
    t2 = clock_type::now();
    double ompMs = chrono::duration<double, milli>(t2 - t1).count();

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    cudaError_t e;
    if ((e = cudaMalloc(&da, n*sizeof(float))) != cudaSuccess ||
        (e = cudaMalloc(&db, n*sizeof(float))) != cudaSuccess ||
        (e = cudaMalloc(&dc, n*sizeof(float))) != cudaSuccess) {
        cerr << "GPU allocation failed: " << cudaGetErrorString(e) << "\n";
        cudaFree(da); cudaFree(db); cudaFree(dc); return 1;
    }
    t1 = clock_type::now();
    cudaMemcpy(da, a.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    int threads = 256, blocks = (n + threads - 1) / threads;
    addGPU<<<blocks, threads>>>(da, db, dc, n);
    e = cudaGetLastError();
    if (e != cudaSuccess) { cerr << cudaGetErrorString(e) << "\n"; return 1; }
    cudaMemcpy(gpu.data(), dc, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    t2 = clock_type::now();
    double cudaTotalMs = chrono::duration<double, milli>(t2 - t1).count();

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (fabs(seq[i] - par[i]) > 1e-5f || fabs(seq[i] - gpu[i]) > 1e-5f) {
            ok = false; break;
        }
    }
    cout << "N=" << n << ", OpenMP max threads=" << omp_get_max_threads() << "\n";
    cout << "Sequential time (ms): " << seqMs << "\n";
    cout << "OpenMP time (ms): " << ompMs << "\n";
    cout << "CUDA total time including transfers (ms): " << cudaTotalMs << "\n";
    cout << "Correctness across all versions: " << (ok ? "PASS" : "FAIL") << "\n";
    cout << "Speedup OpenMP vs sequential: " << seqMs / ompMs << "\n";
    cout << "Speedup CUDA total vs sequential: " << seqMs / cudaTotalMs << "\n";
    cudaFree(da); cudaFree(db); cudaFree(dc);
    return ok ? 0 : 2;
}
