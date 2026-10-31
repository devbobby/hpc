#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double dot_critical(double *a, double *b, long long n)
{
    double sum = 0.0;

    #pragma omp parallel for
    for (long long i = 0; i < n; i++)
    {
        double value = a[i] * b[i];

        #pragma omp critical
        {
            sum += value;
        }
    }

    return sum;
}

double dot_atomic(double *a, double *b, long long n)
{
    double sum = 0.0;

    #pragma omp parallel for
    for (long long i = 0; i < n; i++)
    {
        double value = a[i] * b[i];

        #pragma omp atomic
        sum += value;
    }

    return sum;
}

double dot_reduction(double *a, double *b, long long n)
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

    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));

    for (long long i = 0; i < n; i++)
    {
        a[i] = i % 100;
        b[i] = i % 50;
    }

    double start, end;

    start = omp_get_wtime();
    double c = dot_critical(a, b, n);
    end = omp_get_wtime();
    double tc = end - start;

    start = omp_get_wtime();
    double at = dot_atomic(a, b, n);
    end = omp_get_wtime();
    double ta = end - start;

    start = omp_get_wtime();
    double r = dot_reduction(a, b, n);
    end = omp_get_wtime();
    double tr = end - start;

    printf("N = %lld\n", n);

    printf("\nCritical\n");
    printf("Result = %.2f\n", c);
    printf("Time = %.6f seconds\n", tc);

    printf("\nAtomic\n");
    printf("Result = %.2f\n", at);
    printf("Time = %.6f seconds\n", ta);

    printf("\nReduction\n");
    printf("Result = %.2f\n", r);
    printf("Time = %.6f seconds\n", tr);

    free(a);
    free(b);

    return 0;
}
