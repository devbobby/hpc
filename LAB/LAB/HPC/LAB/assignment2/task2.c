#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main()
{
    long long sizes[] = {10000, 1000000, 10000000};

    for (int s = 0; s < 3; s++)
    {
        long long N = sizes[s];

        double *A = malloc(N * sizeof(double));

        for (long long i = 0; i < N; i++)
            A[i] = 1.0;

        double seq_sum = 0.0;
        double par_sum = 0.0;

        double start = omp_get_wtime();

        for (long long i = 0; i < N; i++)
            seq_sum += A[i];

        double seq_time = omp_get_wtime() - start;

        start = omp_get_wtime();

        #pragma omp parallel for reduction(+:par_sum)
        for (long long i = 0; i < N; i++)
            par_sum += A[i];

        double par_time = omp_get_wtime() - start;

        int threads = omp_get_max_threads();

        double speedup = seq_time / par_time;

        printf("\nN = %lld\n", N);
        printf("Sequential Sum = %.0f\n", seq_sum);
        printf("Parallel Sum = %.0f\n", par_sum);
        printf("Sequential Time = %.6f seconds\n", seq_time);
        printf("Parallel Time = %.6f seconds\n", par_time);
        printf("Threads = %d\n", threads);
        printf("Speedup = %.4f\n", speedup);

        free(A);
    }

    return 0;
}
