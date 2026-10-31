#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void fill(double *M, int N, int seed) {
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            M[i*N+j]=(double)((i+j+seed)%20)/20.0;
}

int main(int argc,char **argv) {
    int rank,size,N=400;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if (argc>1) N=atoi(argv[1]);
    int q=(int)(sqrt((double)size)+0.5);

    if (q*q!=size || N%q!=0) {
        if(rank==0)
            printf("Use a perfect-square process count (4,9,16,...) and N divisible by sqrt(P).\n");
        MPI_Finalize(); return 1;
    }

    int dims[2]={q,q}, periods[2]={0,0}, coords[2];
    MPI_Comm grid,row_comm,col_comm;
    MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,0,&grid);
    MPI_Comm_rank(grid,&rank);
    MPI_Cart_coords(grid,rank,2,coords);

    /* Row communicator: same grid row; column communicator: same grid column. */
    MPI_Comm_split(grid,coords[0],coords[1],&row_comm);
    MPI_Comm_split(grid,coords[1],coords[0],&col_comm);

    int b=N/q;
    double *Ablock=malloc((size_t)b*b*sizeof(double));
    double *Bblock=malloc((size_t)b*b*sizeof(double));
    double *Cblock=malloc((size_t)b*b*sizeof(double));
    double *Atmp=malloc((size_t)b*b*sizeof(double));
    double *Btmp=malloc((size_t)b*b*sizeof(double));

    double *A=NULL,*B=NULL,*C=NULL;
    if(rank==0) {
        A=malloc((size_t)N*N*sizeof(double));
        B=malloc((size_t)N*N*sizeof(double));
        C=malloc((size_t)N*N*sizeof(double));
        fill(A,N,0); fill(B,N,3);
    }

    /* Root sends initial A blocks and B blocks to the correct grid ranks.
       For clarity this uses point-to-point only for the initial distribution;
       the computation then uses collective broadcasts across rows/columns. */
    if(rank==0) {
        for(int pr=0;pr<q;pr++) for(int pc=0;pc<q;pc++) {
            int dest;
            int cc[2]={pr,pc};
            MPI_Cart_rank(grid,cc,&dest);
            double *pa=malloc((size_t)b*b*sizeof(double));
            double *pb=malloc((size_t)b*b*sizeof(double));
            for(int i=0;i<b;i++) for(int j=0;j<b;j++) {
                pa[i*b+j]=A[(pr*b+i)*N+pc*b+j];
                pb[i*b+j]=B[(pr*b+i)*N+pc*b+j];
            }
            if(dest==0) {
                for(int z=0;z<b*b;z++){Ablock[z]=pa[z];Bblock[z]=pb[z];}
            } else {
                MPI_Send(pa,b*b,MPI_DOUBLE,dest,10,grid);
                MPI_Send(pb,b*b,MPI_DOUBLE,dest,11,grid);
            }
            free(pa); free(pb);
        }
    } else {
        MPI_Recv(Ablock,b*b,MPI_DOUBLE,0,10,grid,MPI_STATUS_IGNORE);
        MPI_Recv(Bblock,b*b,MPI_DOUBLE,0,11,grid,MPI_STATUS_IGNORE);
    }

    for(int z=0;z<b*b;z++) Cblock[z]=0.0;

    MPI_Barrier(grid);
    double start=MPI_Wtime();

    for(int k=0;k<q;k++) {
        int root_rank_in_row=k;
        int root_rank_in_col=k;

        /* Broadcast A(i,k) block across row i. */
        if(coords[1]==k)
            for(int z=0;z<b*b;z++) Atmp[z]=Ablock[z];
        MPI_Bcast(Atmp,b*b,MPI_DOUBLE,root_rank_in_row,row_comm);

        /* Broadcast B(k,j) block down column j. */
        if(coords[0]==k)
            for(int z=0;z<b*b;z++) Btmp[z]=Bblock[z];
        MPI_Bcast(Btmp,b*b,MPI_DOUBLE,root_rank_in_col,col_comm);

        for(int i=0;i<b;i++)
            for(int j=0;j<b;j++)
                for(int x=0;x<b;x++)
                    Cblock[i*b+j]+=Atmp[i*b+x]*Btmp[x*b+j];
    }

    double elapsed=MPI_Wtime()-start;
    double max_elapsed;
    MPI_Reduce(&elapsed,&max_elapsed,1,MPI_DOUBLE,MPI_MAX,0,grid);

    /* Gather each C block to root using point-to-point for simple reconstruction. */
    if(rank==0) {
        for(int i=0;i<b;i++) for(int j=0;j<b;j++)
            C[i*N+j]=Cblock[i*b+j];

        for(int pr=0;pr<q;pr++) for(int pc=0;pc<q;pc++) {
            int cc[2]={pr,pc}, src_rank;
            MPI_Cart_rank(grid,cc,&src_rank);
            if(src_rank==0) continue;
            MPI_Recv(Cblock,b*b,MPI_DOUBLE,src_rank,20,grid,MPI_STATUS_IGNORE);
            for(int i=0;i<b;i++) for(int j=0;j<b;j++)
                C[(pr*b+i)*N+pc*b+j]=Cblock[i*b+j];
        }
        printf("2D block matrix multiplication: N=%d P=%d Time=%.8f s\n",
               N,size,max_elapsed);
        free(A);free(B);free(C);
    } else {
        MPI_Send(Cblock,b*b,MPI_DOUBLE,0,20,grid);
    }

    free(Ablock);free(Bblock);free(Cblock);free(Atmp);free(Btmp);
    MPI_Comm_free(&row_comm); MPI_Comm_free(&col_comm); MPI_Comm_free(&grid);
    MPI_Finalize();
    return 0;
}
