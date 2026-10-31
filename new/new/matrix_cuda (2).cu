/*
 * MATRIX ADDITION - CUDA
 *
 * Compile: nvcc matrix_cuda.cu -o matrix_cuda
 * Run:     ./matrix_cuda
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define N 3
#define SIZE (N * N)

__global__ void matrixAdd(const int *A, const int *B, int *C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < SIZE) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    int h_A[SIZE] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    int h_B[SIZE] = {
        10, 20, 30,
        40, 50, 60,
        70, 80, 90
    };

    int h_C[SIZE];

    int *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, SIZE * sizeof(int));
    cudaMalloc((void**)&d_B, SIZE * sizeof(int));
    cudaMalloc((void**)&d_C, SIZE * sizeof(int));

    cudaMemcpy(d_A, h_A, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;

    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result Matrix:\\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_C[i * N + j]);
        }
        printf("\\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
