#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
using namespace std;

int main(int argc,char** argv){
    MPI_Init(&argc,&argv);
    int rank,p;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&p);
    int n=(argc>1)?atoi(argv[1]):1000;
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
    if(n<1){if(rank==0)cerr<<"Size must be positive\n";MPI_Finalize();return 1;}
    vector<int> rows(p),counts(p),displs(p);
    int base=n/p,rem=n%p,off=0;
    for(int r=0;r<p;r++){rows[r]=base+(r<rem?1:0);counts[r]=rows[r]*n;displs[r]=off*n;off+=rows[r];}
    int m=rows[rank], startRow=displs[rank]/n;
    vector<double> full, local((size_t)(m+2)*n,0.0), out((size_t)m*n,0.0);
    if(rank==0){full.resize((size_t)n*n);for(int i=0;i<n;i++)for(int j=0;j<n;j++)full[(size_t)i*n+j]=((i*13+j*5)%251)/251.0;}
    MPI_Barrier(MPI_COMM_WORLD);
    double start=MPI_Wtime();
    MPI_Scatterv(rank==0?full.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,
                 local.data()+n,m*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
    // Exchange first/last owned rows with neighbors to fill halo (ghost) rows.
    if(m>0){
        int prev=(rank==0?MPI_PROC_NULL:rank-1), next=(rank==p-1?MPI_PROC_NULL:rank+1);
        MPI_Sendrecv(local.data()+n,n,MPI_DOUBLE,prev,10,
                     local.data()+(m+1)*n,n,MPI_DOUBLE,next,10,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(local.data()+m*n,n,MPI_DOUBLE,next,20,
                     local.data(),n,MPI_DOUBLE,prev,20,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    const double K[3][3]={{1,1,1},{1,1,1},{1,1,1}};
    for(int li=0;li<m;li++){
        int gi=startRow+li;
        if(gi==0||gi==n-1)continue;
        for(int j=1;j<n-1;j++){
            double v=0;
            for(int u=-1;u<=1;u++)for(int w=-1;w<=1;w++)
                v+=local[(size_t)(li+1+u)*n+(j+w)]*K[u+1][w+1];
            out[(size_t)li*n+j]=v;
        }
    }
    vector<double> result;if(rank==0)result.resize((size_t)n*n);
    MPI_Gatherv(out.data(),m*n,MPI_DOUBLE,rank==0?result.data():nullptr,
                counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    double elapsed=MPI_Wtime()-start,maxElapsed=0;
    MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0){double sum=0;for(double x:result)sum+=x;
        cout<<fixed<<setprecision(6)<<"Application=convolution Size="<<n<<" Model=MPI Workers="<<p
            <<" Time_s="<<maxElapsed<<" Checksum="<<setprecision(10)<<sum<<"\n";}
    MPI_Finalize();
}
