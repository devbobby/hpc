// cuBLAS SGEMM. Row-major C=A*B is obtained by column-major SGEMM B*A.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
using namespace std;
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){cerr<<cudaGetErrorString(e)<<" at line "<<__LINE__<<"\n";return 1;}}while(0)
#define BL(x) do{cublasStatus_t s=(x);if(s!=CUBLAS_STATUS_SUCCESS){cerr<<"cuBLAS error "<<(int)s<<" at line "<<__LINE__<<"\n";return 1;}}while(0)
int main(int argc,char**argv){int N=argc>1?atoi(argv[1]):256;if(N<=0)return 1;int d=0;CK(cudaGetDeviceCount(&d));if(!d){cerr<<"No CUDA GPU found\n";return 2;}vector<float>A((size_t)N*N),B((size_t)N*N),C((size_t)N*N);for(int i=0;i<N*N;i++){A[i]=(i%17)*.1f;B[i]=(i%13)*.2f;}float *a,*b,*c;CK(cudaMalloc(&a,A.size()*4));CK(cudaMalloc(&b,B.size()*4));CK(cudaMalloc(&c,C.size()*4));CK(cudaMemcpy(a,A.data(),A.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(b,B.data(),B.size()*4,cudaMemcpyHostToDevice));cublasHandle_t h;BL(cublasCreate(&h));float alpha=1,beta=0;cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);cudaEventRecord(s);BL(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&alpha,b,N,a,N,&beta,c,N));cudaEventRecord(e);cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,s,e);CK(cudaMemcpy(C.data(),c,C.size()*4,cudaMemcpyDeviceToHost));double sum=0;for(float x:C)sum+=x;cout<<fixed<<setprecision(4)<<"N="<<N<<", cuBLAS SGEMM ms="<<ms<<", checksum="<<sum<<"\n";cudaEventDestroy(s);cudaEventDestroy(e);cublasDestroy(h);cudaFree(a);cudaFree(b);cudaFree(c);}
