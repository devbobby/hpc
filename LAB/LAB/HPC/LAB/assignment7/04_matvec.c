#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void fill_data(double *A, double *x, int N) {
    for (int i = 0; i < N; i++) {
        x[i] = 1.0;
        for (int j = 0; j < N; j++)
            A[i*N + j] = (double)((i + j) % 100) / 100.0;
    }
}

static double sequential_matvec(const double *A, const double *x, double *y, int N) {
    double start = MPI_Wtime();
    for (int i = 0; i < N; i++) {
        y[i] = 0.0;
        for (int j = 0; j < N; j++)
            y[i] += A[i*N + j] * x[j];
    }
    return MPI_Wtime() - start;
}

int main(int argc, char *argv[]) {
    int rank, size, N = 1000;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc >= 2) N = atoi(argv[1]);
    if (N <= 0 || N % size != 0) {
        if (rank == 0)
            printf("Usage: mpirun -np P ./matvec N, where N is divisible by P.\n");
        MPI_Finalize();
        return 1;
    }

    int rows = N / size;
    double *A = NULL, *x = NULL, *y = NULL, *y_seq = NULL;
    double *local_A = (double *)malloc((size_t)rows * N * sizeof(double));
    double *local_y = (double *)malloc(rows * sizeof(double));

    if (rank == 0) {
        A = (double *)malloc((size_t)N * N * sizeof(double));
        x = (double *)malloc(N * sizeof(double));
        y = (double *)malloc(N * sizeof(double));
        y_seq = (double *)malloc(N * sizeof(double));
        fill_data(A, x, N);
        double seq_time = sequential_matvec(A, x, y_seq, N);
        printf("N=%d Sequential Time=%.8f s\n", N, seq_time);
    } else {
        x = (double *)malloc(N * sizeof(double));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatter(A, rows*N, MPI_DOUBLE,
                local_A, rows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows; i++) {
        local_y[i] = 0.0;
        for (int j = 0; j < N; j++)
            local_y[i] += local_A[i*N + j] * x[j];
    }

    MPI_Gather(local_y, rows, MPI_DOUBLE,
               y, rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double local_time = MPI_Wtime() - start;
    double parallel_time;
    MPI_Reduce(&local_time, &parallel_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("N=%d Processes=%d MPI Time=%.8f s\n", N, size, parallel_time);
        if (y_seq) {
            double max_err = 0.0;
            for (int i = 0; i < N; i++) {
                double err = y[i] - y_seq[i];
                if (err < 0) err = -err;
                if (err > max_err) max_err = err;
            }
            printf("Maximum verification error = %.12f\n", max_err);
        }
    }

    free(local_A);
    free(local_y);
    free(x);
    if (rank == 0) {
        free(A); free(y); free(y_seq);
    }

    MPI_Finalize();
    return 0;
}
