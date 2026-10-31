#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double sequential_dot(double *a, double *b, long long n)
{
    double sum = 0.0;

    for (long long i = 0; i < n; i++)
        sum += a[i] * b[i];

    return sum;
}

double parallel_dot(double *a, double *b, long long n)
{
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < n; i++)
        sum += a[i] * b[i];

    return sum;
}

int main()
{
    long long n = 10000000;

    double *a = (double *)malloc(n * sizeof(double));
    double *b = (double *)malloc(n * sizeof(double));

    for (long long i = 0; i < n; i++)
    {
        a[i] = (double)(i % 100);
        b[i] = (double)(i % 50);
    }

    double start, end;

    start = omp_get_wtime();
    double sequential_result = sequential_dot(a, b, n);
    end = omp_get_wtime();

    double sequential_time = end - start;

    start = omp_get_wtime();
    double parallel_result = parallel_dot(a, b, n);
    end = omp_get_wtime();

    double parallel_time = end - start;

    printf("Array Size: %lld\n", n);

    printf("Sequential Dot Product = %.2f\n", sequential_result);
    printf("Sequential Time = %.6f seconds\n", sequential_time);

    printf("Parallel Dot Product = %.2f\n", parallel_result);
    printf("Parallel Time = %.6f seconds\n", parallel_time);

    printf("Threads = %d\n", omp_get_max_threads());

    printf("Speedup = %.4f\n", sequential_time / parallel_time);

    printf("Efficiency = %.2f%%\n",
           (sequential_time / parallel_time) /
           omp_get_max_threads() * 100.0);

    free(a);
    free(b);

    return 0;
}
