#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double sequential_sum(double *a, long long n)
{
    double sum = 0.0;

    for (long long i = 0; i < n; i++)
        sum += a[i];

    return sum;
}

double critical_sum(double *a, long long n)
{
    double sum = 0.0;

    #pragma omp parallel for
    for (long long i = 0; i < n; i++)
    {
        #pragma omp critical
        {
            sum += a[i];
        }
    }

    return sum;
}

double reduction_sum(double *a, long long n)
{
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < n; i++)
        sum += a[i];

    return sum;
}

int main()
{
    long long sizes[] = {10000, 1000000, 10000000};
    int num_sizes = 3;

    int threads = omp_get_max_threads();

    printf("Maximum OpenMP Threads: %d\n\n", threads);

    for (int s = 0; s < num_sizes; s++)
    {
        long long n = sizes[s];

        double *a = (double *)malloc(n * sizeof(double));

        for (long long i = 0; i < n; i++)
            a[i] = 1.0;

        double start, end;
        double seq_sum, critical_result, reduction_result;
        double seq_time, critical_time, reduction_time;

        start = omp_get_wtime();
        seq_sum = sequential_sum(a, n);
        end = omp_get_wtime();
        seq_time = end - start;

        start = omp_get_wtime();
        critical_result = critical_sum(a, n);
        end = omp_get_wtime();
        critical_time = end - start;

        start = omp_get_wtime();
        reduction_result = reduction_sum(a, n);
        end = omp_get_wtime();
        reduction_time = end - start;

        printf("Array Size: %lld\n", n);

        printf("Sequential:\n");
        printf("Sum = %.2f\n", seq_sum);
        printf("Time = %.6f seconds\n", seq_time);

        printf("Critical:\n");
        printf("Sum = %.2f\n", critical_result);
        printf("Time = %.6f seconds\n", critical_time);

        printf("Reduction:\n");
        printf("Sum = %.2f\n", reduction_result);
        printf("Time = %.6f seconds\n", reduction_time);

        printf("----------------------------------------\n");

        free(a);
    }

    return 0;
}
