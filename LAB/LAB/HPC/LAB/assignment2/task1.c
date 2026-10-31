#include <stdio.h>
#include <omp.h>

int main()
{
    int thread_count[] = {2, 4, 8};

    for (int k = 0; k < 3; k++)
    {
        omp_set_num_threads(thread_count[k]);

        printf("\nRequested Threads: %d\n", thread_count[k]);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int total = omp_get_num_threads();

            printf("Thread ID: %d, Total Threads: %d\n", tid, total);
        }
    }

    return 0;
}
