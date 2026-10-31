/*
 * DOT PRODUCT - MPI
 *
 * Compile: mpicc dot_mpi.c -o dot_mpi
 * Run:     mpirun -np 4 ./dot_mpi
 *
 * This example uses 8 elements and 4 MPI processes.
 */

#include <stdio.h>
#include <mpi.h>

#define N 8

int main(int argc, char *argv[]) {
    int A[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    int B[N] = {2, 3, 4, 5, 6, 7, 8, 9};

    int rank, size;
    int local_n;
    int local_A[N / 4], local_B[N / 4];
    int local_dot = 0;
    int global_dot = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0) {
            printf("Please run with exactly 4 processes.\\n");
        }
        MPI_Finalize();
        return 0;
    }

    local_n = N / size;

    MPI_Scatter(A, local_n, MPI_INT,
                local_A, local_n, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Scatter(B, local_n, MPI_INT,
                local_B, local_n, MPI_INT,
                0, MPI_COMM_WORLD);

    for (int i = 0; i < local_n; i++) {
        local_dot += local_A[i] * local_B[i];
    }

    MPI_Reduce(&local_dot, &global_dot, 1, MPI_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Dot Product = %d\\n", global_dot);
    }

    MPI_Finalize();
    return 0;
}
