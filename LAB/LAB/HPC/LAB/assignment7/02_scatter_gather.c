#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 16

int main(int argc, char *argv[]) {
    int rank, size;
    int data[N];
    int local_count, local_sum = 0;
    int *local = NULL, *sums = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0)
            printf("Use a process count that divides %d (e.g., 2, 4, 8).\n", N);
        MPI_Finalize();
        return 1;
    }

    local_count = N / size;
    local = (int *)malloc(local_count * sizeof(int));

    if (rank == 0) {
        for (int i = 0; i < N; i++)
            data[i] = i + 1;
        sums = (int *)malloc(size * sizeof(int));
    }

    MPI_Scatter(data, local_count, MPI_INT,
                local, local_count, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_count; i++)
        local_sum += local[i];

    printf("Process %d local sum = %d\n", rank, local_sum);

    MPI_Gather(&local_sum, 1, MPI_INT,
               sums, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int final_sum = 0;
        printf("\nGathered local sums:\n");
        for (int i = 0; i < size; i++) {
            printf("Process %d -> %d\n", i, sums[i]);
            final_sum += sums[i];
        }
        printf("Final sum = %d\n", final_sum);
        free(sums);
    }

    free(local);
    MPI_Finalize();
    return 0;
}
