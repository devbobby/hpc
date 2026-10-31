/*
 * MATRIX ADDITION - MPI
 *
 * Compile: mpicc matrix_mpi.c -o matrix_mpi
 * Run:     mpirun -np 3 ./matrix_mpi
 *
 * This example uses 3 processes. Each process receives one row.
 */

#include <stdio.h>
#include <mpi.h>

#define N 3

int main(int argc, char *argv[]) {
    int A[N][N] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    int B[N][N] = {
        {10, 20, 30},
        {40, 50, 60},
        {70, 80, 90}
    };

    int C[N][N];
    int local_A[N], local_B[N], local_C[N];
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != N) {
        if (rank == 0) {
            printf("Please run with exactly 3 processes.\\n");
        }
        MPI_Finalize();
        return 0;
    }

    MPI_Scatter(A, N, MPI_INT, local_A, N, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Scatter(B, N, MPI_INT, local_B, N, MPI_INT,
                0, MPI_COMM_WORLD);

    for (int j = 0; j < N; j++) {
        local_C[j] = local_A[j] + local_B[j];
    }

    MPI_Gather(local_C, N, MPI_INT, C, N, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Result Matrix:\\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", C[i][j]);
            }
            printf("\\n");
        }
    }

    MPI_Finalize();
    return 0;
}
