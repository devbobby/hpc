#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void sequential_scan(long long *input, long long *output, long long n)
{
    if (n <= 0)
        return;

    output[0] = input[0];

    for (long long i = 1; i < n; i++)
        output[i] = output[i - 1] + input[i];
}

void parallel_scan(long long *input, long long *output, long long n)
{
    int p = omp_get_max_threads();

    if (p > n)
        p = (int)n;

    long long *thread_sum =
        (long long *)calloc(p, sizeof(long long));

    #pragma omp parallel num_threads(p)
    {
        int tid = omp_get_thread_num();

        long long start = (n * tid) / p;
        long long end = (n * (tid + 1)) / p;

        if (start < end)
        {
            output[start] = input[start];

            for (long long i = start + 1; i < end; i++)
                output[i] = output[i - 1] + input[i];

            thread_sum[tid] = output[end - 1];
        }
        else
        {
            thread_sum[tid] = 0;
        }

        #pragma omp barrier

        #pragma omp single
        {
            for (int i = 1; i < p; i++)
                thread_sum[i] += thread_sum[i - 1];
        }

        #pragma omp barrier

        if (tid > 0 && start < end)
        {
            long long offset = thread_sum[tid - 1];

            for (long long i = start; i < end; i++)
                output[i] += offset;
        }
    }

    free(thread_sum);
}

int main()
{
    long long n = 10000000;

    long long *input =
        (long long *)malloc(n * sizeof(long long));

    long long *seq_output =
        (long long *)malloc(n * sizeof(long long));

    long long *parallel_output =
        (long long *)malloc(n * sizeof(long long));

    for (long long i = 0; i < n; i++)
        input[i] = (i % 100) + 1;

    double start, end;

    start = omp_get_wtime();

    sequential_scan(input, seq_output, n);

    end = omp_get_wtime();

    double sequential_time = end - start;

    start = omp_get_wtime();

    parallel_scan(input, parallel_output, n);

    end = omp_get_wtime();

    double parallel_time = end - start;

    int correct = 1;

    for (long long i = 0; i < n; i++)
    {
        if (seq_output[i] != parallel_output[i])
        {
            correct = 0;
            break;
        }
    }

    printf("Array Size: %lld\n", n);

    printf("Sequential Time = %.6f seconds\n",
           sequential_time);

    printf("Parallel Time = %.6f seconds\n",
           parallel_time);

    printf("Threads = %d\n", omp_get_max_threads());

    printf("Speedup = %.4f\n",
           sequential_time / parallel_time);

    printf("Efficiency = %.2f%%\n",
           (sequential_time / parallel_time) /
           omp_get_max_threads() * 100.0);

    printf("Correctness = %s\n",
           correct ? "PASS" : "FAIL");

    printf("Last Sequential Value = %lld\n",
           seq_output[n - 1]);

    printf("Last Parallel Value = %lld\n",
           parallel_output[n - 1]);

    free(input);
    free(seq_output);
    free(parallel_output);

    return 0;
}
