#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void multiply_static(double *A, double *B, double *C, int N)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            double sum = 0.0;

            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];

            C[i * N + j] = sum;
        }
}

void multiply_dynamic(double *A, double *B, double *C, int N)
{
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            double sum = 0.0;

            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];

            C[i * N + j] = sum;
        }
}

void multiply_guided(double *A, double *B, double *C, int N)
{
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            double sum = 0.0;

            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];

            C[i * N + j] = sum;
        }
}

int main()
{
    int N = 1000;

    double *A = malloc((long long)N * N * sizeof(double));
    double *B = malloc((long long)N * N * sizeof(double));
    double *C = malloc((long long)N * N * sizeof(double));

    for (long long i = 0; i < (long long)N * N; i++)
    {
        A[i] = 1.0;
        B[i] = 1.0;
    }

    double start, time;

    start = omp_get_wtime();
    multiply_static(A, B, C, N);
    time = omp_get_wtime() - start;
    printf("Static: %.6f seconds\n", time);

    start = omp_get_wtime();
    multiply_dynamic(A, B, C, N);
    time = omp_get_wtime() - start;
    printf("Dynamic: %.6f seconds\n", time);

    start = omp_get_wtime();
    multiply_guided(A, B, C, N);
    time = omp_get_wtime() - start;
    printf("Guided: %.6f seconds\n", time);

    free(A);
    free(B);
    free(C);

    return 0;
}
