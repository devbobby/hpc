// Basic CUDA: one thread computes one output element.
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
using namespace std;
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){cerr<<cudaGetErrorString(e)<<" at line "<<__LINE__<<"\n";return 1;}}while(0)
__global__ void basic(const float*A,const float*B,float*C,int N){int col=blockIdx.x*blockDim.x+threadIdx.x,row=blockIdx.y*blockDim.y+threadIdx.y;if(row<N&&col<N){float sum=0;for(int k=0;k<N;k++)sum+=A[row*N+k]*B[k*N+col];C[row*N+col]=sum;}}
int main(int argc,char**argv){int N=argc>1?atoi(argv[1]):256,T=argc>2?atoi(argv[2]):16;if(N<=0||(T!=8&&T!=16&&T!=32))return 1;int d=0;CK(cudaGetDeviceCount(&d));if(!d){cerr<<"No CUDA GPU found\n";return 2;}vector<float>A((size_t)N*N),B((size_t)N*N),C((size_t)N*N);for(int i=0;i<N*N;i++){A[i]=(i%17)*.1f;B[i]=(i%13)*.2f;}float *a,*b,*c;CK(cudaMalloc(&a,A.size()*4));CK(cudaMalloc(&b,B.size()*4));CK(cudaMalloc(&c,C.size()*4));CK(cudaMemcpy(a,A.data(),A.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(b,B.data(),B.size()*4,cudaMemcpyHostToDevice));dim3 block(T,T),grid((N+T-1)/T,(N+T-1)/T);cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);cudaEventRecord(s);basic<<<grid,block>>>(a,b,c,N);CK(cudaGetLastError());cudaEventRecord(e);cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,s,e);CK(cudaMemcpy(C.data(),c,C.size()*4,cudaMemcpyDeviceToHost));double sum=0;for(float x:C)sum+=x;cout<<fixed<<setprecision(4)<<"N="<<N<<", tile="<<T<<", basic CUDA kernel ms="<<ms<<", checksum="<<sum<<"\n";cudaFree(a);cudaFree(b);cudaFree(c);cudaEventDestroy(s);cudaEventDestroy(e);}
