#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, p; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&p);
    int n = (argc > 1) ? atoi(argv[1]) : 500;
    int bad = (n < 1);
    MPI_Bcast(&bad,1,MPI_INT,0,MPI_COMM_WORLD);
    if (bad) { if(rank==0) cerr<<"Size must be positive\n"; MPI_Finalize(); return 1; }
    // Broadcast the requested size to ensure every rank uses the same value.
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
    vector<int> rows(p), counts(p), displs(p), elemcounts(p), elemdispls(p);
    int base=n/p, rem=n%p, offset=0;
    for(int r=0;r<p;r++) {
        rows[r]=base+(r<rem?1:0);
        counts[r]=rows[r]*n; displs[r]=offset*n; offset+=rows[r];
        elemcounts[r]=counts[r]; elemdispls[r]=displs[r];
    }
    int localRows=rows[rank];
    vector<double> A, B((size_t)n*n), localA((size_t)localRows*n), localC((size_t)localRows*n,0.0);
    if(rank==0) {
        A.resize((size_t)n*n);
        for(int i=0;i<n;i++) for(int j=0;j<n;j++) {
            A[(size_t)i*n+j]=((i*17+j*3)%101)/101.0;
            B[(size_t)i*n+j]=((i*7+j*11)%97)/97.0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start=MPI_Wtime();
    MPI_Scatterv(rank==0?A.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,
                 localA.data(),localRows*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(B.data(),n*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
    for(int i=0;i<localRows;i++)
        for(int k=0;k<n;k++) {
            double a=localA[(size_t)i*n+k];
            for(int j=0;j<n;j++) localC[(size_t)i*n+j]+=a*B[(size_t)k*n+j];
        }
    vector<double> C;
    if(rank==0) C.resize((size_t)n*n);
    MPI_Gatherv(localC.data(),localRows*n,MPI_DOUBLE,
                rank==0?C.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    double elapsed=MPI_Wtime()-start, maxElapsed=0;
    MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0) {
        double sum=0; for(double x:C) sum+=x;
        cout<<fixed<<setprecision(6)<<"Application=matrix Size="<<n<<" Model=MPI Workers="<<p
            <<" Time_s="<<maxElapsed<<" Checksum="<<setprecision(10)<<sum<<"\n";
    }
    MPI_Finalize();
}
