/*
 * Matrix Addition: Sequential + OpenMP + MPI
 *
 * Save as: matrix_all.c
 *
 * Compile sequential:
 *   gcc matrix_all.c -o matrix_seq
 *
 * Compile OpenMP:
 *   gcc -fopenmp matrix_all.c -o matrix_omp
 *
 * Compile MPI:
 *   mpicc matrix_all.c -o matrix_mpi
 *
 * Run:
 *   ./matrix_seq
 *   OMP_NUM_THREADS=4 ./matrix_omp
 *   mpirun -np 4 ./matrix_mpi
 *
 * Select implementation at compile time:
 *   -DOPENMP for OpenMP
 *   -DMPI_VERSION for MPI
 */

#include <stdio.h>

#define N 3

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

void print_matrix(void)
{
    printf("Result Matrix:\n");

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }
}

/* 1. Sequential matrix addition */
void sequential_add(void)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

/* 2. OpenMP matrix addition */
#ifdef OPENMP
#include <omp.h>

void openmp_add(void)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}
#endif

/* 3. MPI matrix addition: rank 0 distributes elements */
#ifdef MPI_VERSION
#include <mpi.h>

void mpi_add(int rank, int size)
{
    if (size != N * N) {
        if (rank == 0) {
            printf("Please run with %d processes: mpirun -np %d ./matrix_mpi\n",
                   N * N, N * N);
        }
        return;
    }

    if (rank == 0) {
        C[0][0] = A[0][0] + B[0][0];

        for (int i = 1; i < N * N; i++) {
            int row = i / N;
            int col = i % N;

            MPI_Send(&A[row][col], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&B[row][col], 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        }

        for (int i = 1; i < N * N; i++) {
            int row = i / N;
            int col = i % N;

            MPI_Recv(&C[row][col], 1, MPI_INT, i, 2,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        int x, y, result;

        MPI_Recv(&x, 1, MPI_INT, 0, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&y, 1, MPI_INT, 0, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        result = x + y;

        MPI_Send(&result, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }
}
#endif

int main(int argc, char *argv[])
{
#ifdef MPI_VERSION
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    mpi_add(rank, size);

    if (rank == 0) {
        print_matrix();
    }

    MPI_Finalize();

#elif defined(OPENMP)
    openmp_add();
    print_matrix();

#else
    sequential_add();
    print_matrix();
#endif

    return 0;
}
