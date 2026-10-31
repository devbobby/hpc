/*
 * MATRIX ADDITION - OPENMP
 *
 * Compile: gcc -fopenmp matrix_openmp.c -o matrix_openmp
 * Run:     ./matrix_openmp
 * Optional: OMP_NUM_THREADS=4 ./matrix_openmp
 */

#include <stdio.h>
#include <omp.h>

#define N 3

int main(void) {
    int A[N][N] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    int B[N][N] = {
        {10, 20, 30},
        {40, 50, 60},
        {70, 80, 90}
    };

    int C[N][N];

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }

    printf("Result Matrix:\\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", C[i][j]);
        }
        printf("\\n");
    }

    return 0;
}
