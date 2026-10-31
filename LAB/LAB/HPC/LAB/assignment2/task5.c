#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void multiply(double *A, double *B, double *C, int N)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double sum = 0.0;

            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];

            C[i * N + j] = sum;
        }
    }
}

int main()
{
    int N = 2000;

    double *A = malloc((long long)N * N * sizeof(double));
    double *B = malloc((long long)N * N * sizeof(double));
    double *C = malloc((long long)N * N * sizeof(double));

    for (long long i = 0; i < (long long)N * N; i++)
    {
        A[i] = 1.0;
        B[i] = 1.0;
        C[i] = 0.0;
    }

    double start = omp_get_wtime();

    multiply(A, B, C, N);

    double time = omp_get_wtime() - start;

    printf("Matrix Size: %d x %d\n", N, N);
    printf("Threads: %d\n", omp_get_max_threads());
    printf("Execution Time: %.6f seconds\n", time);
    printf("C[0] = %.2f\n", C[0]);

    free(A);
    free(B);
    free(C);

    return 0;
}
