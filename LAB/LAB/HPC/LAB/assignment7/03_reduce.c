#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 16

int main(int argc, char *argv[]) {
    int rank, size;
    int data[N];
    int local_count;
    int *local;
    int local_sum = 0, local_max, local_min;
    int global_sum, global_max, global_min;

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
    }

    MPI_Scatter(data, local_count, MPI_INT,
                local, local_count, MPI_INT, 0, MPI_COMM_WORLD);

    local_max = local[0];
    local_min = local[0];

    for (int i = 0; i < local_count; i++) {
        local_sum += local[i];
        if (local[i] > local_max) local_max = local[i];
        if (local[i] < local_min) local_min = local[i];
    }

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    printf("Process %d: local sum=%d, local min=%d, local max=%d\n",
           rank, local_sum, local_min, local_max);

    if (rank == 0) {
        printf("\nGlobal sum = %d\n", global_sum);
        printf("Global minimum = %d\n", global_min);
        printf("Global maximum = %d\n", global_max);
    }

    free(local);
    MPI_Finalize();
    return 0;
}
