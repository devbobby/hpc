#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void fill_matrix(double *M, int N, int seed) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            M[i*N + j] = (double)((i + j + seed) % 50) / 50.0;
}

static double sequential_matmat(const double *A, const double *B, double *C, int N) {
    double start = MPI_Wtime();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i*N + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
    }
    return MPI_Wtime() - start;
}

int main(int argc, char *argv[]) {
    int rank, size, N = 500;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc >= 2) N = atoi(argv[1]);
    if (N <= 0 || N % size != 0) {
        if (rank == 0)
            printf("Usage: mpirun -np P ./matmat N, where N is divisible by P.\n");
        MPI_Finalize();
        return 1;
    }

    int rows = N / size;
    double *A = NULL, *B = NULL, *C = NULL, *C_seq = NULL;
    double *local_A = (double *)malloc((size_t)rows * N * sizeof(double));
    double *local_C = (double *)malloc((size_t)rows * N * sizeof(double));

    if (rank == 0) {
        A = (double *)malloc((size_t)N * N * sizeof(double));
        B = (double *)malloc((size_t)N * N * sizeof(double));
        C = (double *)malloc((size_t)N * N * sizeof(double));
        C_seq = (double *)malloc((size_t)N * N * sizeof(double));
        fill_matrix(A, N, 0);
        fill_matrix(B, N, 7);
        double seq_time = sequential_matmat(A, B, C_seq, N);
        printf("N=%d Sequential Time=%.8f s\n", N, seq_time);
    } else {
        B = (double *)malloc((size_t)N * N * sizeof(double));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatter(A, rows*N, MPI_DOUBLE,
                local_A, rows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += local_A[i*N + k] * B[k*N + j];
            local_C[i*N + j] = sum;
        }
    }

    MPI_Gather(local_C, rows*N, MPI_DOUBLE,
               C, rows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double local_time = MPI_Wtime() - start;
    double parallel_time;
    MPI_Reduce(&local_time, &parallel_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("N=%d Processes=%d MPI Time=%.8f s\n", N, size, parallel_time);
        double max_err = 0.0;
        for (int i = 0; i < N*N; i++) {
            double err = C[i] - C_seq[i];
            if (err < 0) err = -err;
            if (err > max_err) max_err = err;
        }
        printf("Maximum verification error = %.12f\n", max_err);
    }

    free(local_A); free(local_C); free(B);
    if (rank == 0) {
        free(A); free(C); free(C_seq);
    }

    MPI_Finalize();
    return 0;
}
