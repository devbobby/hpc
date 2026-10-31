// Sequential CPU matrix multiplication: C = A * B
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
using namespace std;
int main(int argc,char**argv){
 int N=(argc>1)?atoi(argv[1]):256; if(N<=0)return 1;
 vector<float>A((size_t)N*N),B((size_t)N*N),C((size_t)N*N,0);
 for(int i=0;i<N*N;i++){A[i]=(i%17)*0.1f;B[i]=(i%13)*0.2f;}
 auto s=chrono::high_resolution_clock::now();
 for(int i=0;i<N;i++)for(int j=0;j<N;j++){float sum=0;for(int k=0;k<N;k++)sum+=A[(size_t)i*N+k]*B[(size_t)k*N+j];C[(size_t)i*N+j]=sum;}
 auto e=chrono::high_resolution_clock::now();double ms=chrono::duration<double,milli>(e-s).count();double check=0;for(float x:C)check+=x;
 cout<<fixed<<setprecision(4)<<"N="<<N<<", CPU time (ms)="<<ms<<", checksum="<<check<<"\n";
}
