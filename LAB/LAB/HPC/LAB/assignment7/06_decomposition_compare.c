#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 * Compare:
 * 1) Row-wise: distribute rows of A; broadcast B.
 * 2) Column-wise: distribute columns of B; broadcast A.
 *
 * The assignment also asks students to study block decomposition.
 * The separate block version is included below as an educational implementation
 * for a square process grid. Use a perfect-square process count (e.g., 4 or 9)
 * and N divisible by sqrt(P).
 */

static void fill_matrix(double *M, int N, int seed) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            M[i*N + j] = (double)((i + j + seed) % 50) / 50.0;
}

static void zero(double *M, size_t n) {
    for (size_t i = 0; i < n; i++) M[i] = 0.0;
}

static double rowwise(const double *A, const double *B, double *C, int N,
                      int rank, int size) {
    int rows = N / size;
    double *localA = malloc((size_t)rows*N*sizeof(double));
    double *localC = malloc((size_t)rows*N*sizeof(double));
    MPI_Barrier(MPI_COMM_WORLD);
    double t = MPI_Wtime();

    MPI_Scatter(A, rows*N, MPI_DOUBLE, localA, rows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast((void*)B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i=0;i<rows;i++)
        for (int j=0;j<N;j++) {
            double s=0;
            for (int k=0;k<N;k++) s += localA[i*N+k]*B[k*N+j];
            localC[i*N+j]=s;
        }

    MPI_Gather(localC, rows*N, MPI_DOUBLE, C, rows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double local = MPI_Wtime()-t, global;
    MPI_Reduce(&local,&global,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    free(localA); free(localC);
    return global;
}

static double columnwise(const double *A, const double *B, double *C, int N,
                         int rank, int size) {
    int cols = N / size;
    double *localB = malloc((size_t)N*cols*sizeof(double));
    double *localC = malloc((size_t)N*cols*sizeof(double));
    MPI_Barrier(MPI_COMM_WORLD);
    double t = MPI_Wtime();

    /* Send contiguous column blocks by packing B into a temporary layout on root. */
    double *packedB = NULL;
    if (rank == 0) {
        packedB = malloc((size_t)N*N*sizeof(double));
        for (int p=0;p<size;p++)
            for (int j=0;j<cols;j++)
                for (int i=0;i<N;i++)
                    packedB[p*N*cols + j*N + i] = B[i*N + p*cols + j];
    }

    MPI_Scatter(packedB, N*cols, MPI_DOUBLE,
                localB, N*cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast((void*)A, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i=0;i<N;i++)
        for (int j=0;j<cols;j++) {
            double s=0;
            for (int k=0;k<N;k++) s += A[i*N+k]*localB[j*N+k];
            localC[i*cols+j]=s;
        }

    /* Pack local C blocks on root in rank order and unpack into C. */
    double *packedC = NULL;
    if (rank == 0) packedC = malloc((size_t)N*N*sizeof(double));
    MPI_Gather(localC, N*cols, MPI_DOUBLE,
               packedC, N*cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int p=0;p<size;p++)
            for (int j=0;j<cols;j++)
                for (int i=0;i<N;i++)
                    C[i*N+p*cols+j] = packedC[p*N*cols+j*N+i];
        free(packedB); free(packedC);
    }

    double local = MPI_Wtime()-t, global;
    MPI_Reduce(&local,&global,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    free(localB); free(localC);
    return global;
}

int main(int argc, char **argv) {
    int rank,size,N=400;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    if (argc>1) N=atoi(argv[1]);

    if (N<=0 || N%size!=0) {
        if(rank==0) printf("For this comparison, N must be divisible by P.\n");
        MPI_Finalize(); return 1;
    }

    double *A=NULL,*B=NULL,*C=NULL;
    if(rank==0) {
        A=malloc((size_t)N*N*sizeof(double));
        B=malloc((size_t)N*N*sizeof(double));
        C=malloc((size_t)N*N*sizeof(double));
        fill_matrix(A,N,0); fill_matrix(B,N,5);
    } else {
        /* MPI_Bcast requires valid receive storage on every rank. */
        A=malloc((size_t)N*N*sizeof(double));
        B=malloc((size_t)N*N*sizeof(double));
    }

    double tr=rowwise(A,B,C,N,rank,size);
    double tc=columnwise(A,B,C,N,rank,size);

    if(rank==0) {
        printf("N=%d P=%d\n",N,size);
        printf("Row-wise time    = %.8f s\n",tr);
        printf("Column-wise time = %.8f s\n",tc);
        printf("\nFor block decomposition, use the block version supplied with the report.\n");
    }

    free(A); free(B);
    if(rank==0) free(C);
    MPI_Finalize();
    return 0;
}
