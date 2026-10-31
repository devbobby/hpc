#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc,char **argv) {
    int rank,size,N=10;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if(argc>1) N=atoi(argv[1]);
    if(N<=0) {
        MPI_Finalize(); return 1;
    }

    int base=N/size, rem=N%size;
    int local_rows=base+(rank<rem ? 1:0);

    int *counts=malloc(size*sizeof(int));
    int *displs=malloc(size*sizeof(int));
    int offset=0;
    for(int p=0;p<size;p++) {
        int r=base+(p<rem ? 1:0);
        counts[p]=r*N;
        displs[p]=offset;
        offset+=r*N;
    }

    double *A=NULL,*x=malloc((size_t)N*sizeof(double));
    double *y=NULL;
    if(rank==0) {
        A=malloc((size_t)N*N*sizeof(double));
        y=malloc((size_t)N*sizeof(double));
        for(int i=0;i<N;i++) {
            x[i]=1.0;
            for(int j=0;j<N;j++)
                A[i*N+j]=(double)((i+j)%10);
        }
    }

    double *localA=malloc((size_t)local_rows*N*sizeof(double));
    double *localY=malloc((size_t)local_rows*sizeof(double));

    MPI_Bcast(x,N,MPI_DOUBLE,0,MPI_COMM_WORLD);

    MPI_Scatterv(A,counts,displs,MPI_DOUBLE,
                 localA,local_rows*N,MPI_DOUBLE,0,MPI_COMM_WORLD);

    for(int i=0;i<local_rows;i++) {
        localY[i]=0.0;
        for(int j=0;j<N;j++)
            localY[i]+=localA[i*N+j]*x[j];
    }

    int *ycounts=NULL,*ydispls=NULL;
    if(rank==0) {
        ycounts=malloc(size*sizeof(int));
        ydispls=malloc(size*sizeof(int));
        int off=0;
        for(int p=0;p<size;p++) {
            int r=base+(p<rem ? 1:0);
            ycounts[p]=r;
            ydispls[p]=off;
            off+=r;
        }
    }

    MPI_Gatherv(localY,local_rows,MPI_DOUBLE,
                y,ycounts,ydispls,MPI_DOUBLE,0,MPI_COMM_WORLD);

    if(rank==0) {
        printf("MPI_Scatterv matrix-vector multiplication\n");
        printf("N=%d Processes=%d\n",N,size);
        printf("Distribution: ");
        for(int p=0;p<size;p++) printf("P%d=%d rows%s",p,
            base+(p<rem ? 1:0), p==size-1?"\n":", ");
        printf("Result vector Y:\n");
        for(int i=0;i<N;i++) printf("%.2f ",y[i]);
        printf("\n");
        free(A);free(y);free(ycounts);free(ydispls);
    }

    free(x);free(localA);free(localY);free(counts);free(displs);
    MPI_Finalize();
    return 0;
}
