/*
 * DOT PRODUCT - CUDA
 *
 * Compile: nvcc dot_cuda.cu -o dot_cuda
 * Run:     ./dot_cuda
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define N 8

__global__ void dotProduct(const int *A, const int *B, int *partial) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        partial[i] = A[i] * B[i];
    }
}

int main(void) {
    int h_A[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    int h_B[N] = {2, 3, 4, 5, 6, 7, 8, 9};
    int h_partial[N] = {0};
    int dot = 0;

    int *d_A, *d_B, *d_partial;

    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_partial, N * sizeof(int));

    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

    dotProduct<<<1, N>>>(d_A, d_B, d_partial);
    cudaDeviceSynchronize();

    cudaMemcpy(h_partial, d_partial, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        dot += h_partial[i];
    }

    printf("Dot Product = %d\\n", dot);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_partial);

    return 0;
}
