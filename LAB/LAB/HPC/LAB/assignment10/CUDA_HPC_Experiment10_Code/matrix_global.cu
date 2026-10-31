#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<"\n"; std::exit(EXIT_FAILURE);} } while(0)
__global__ void matMulGlobal(const float* A,const float* B,float* C,int N){
 int row=blockIdx.y*blockDim.y+threadIdx.y, col=blockIdx.x*blockDim.x+threadIdx.x;
 if(row<N && col<N){float sum=0; for(int k=0;k<N;++k) sum+=A[row*N+k]*B[k*N+col]; C[row*N+col]=sum;}
}
int main(){
 const int N=128; size_t bytes=(size_t)N*N*sizeof(float);
 std::vector<float>A(N*N),B(N*N),C(N*N),ref(N*N,0);
 for(int i=0;i<N*N;++i){A[i]=(i%13+1)/13.0f; B[i]=(i%7+1)/7.0f;}
 for(int r=0;r<N;++r)for(int c=0;c<N;++c)for(int k=0;k<N;++k)ref[r*N+c]+=A[r*N+k]*B[k*N+c];
 float *dA,*dB,*dC; CUDA_CHECK(cudaMalloc(&dA,bytes)); CUDA_CHECK(cudaMalloc(&dB,bytes)); CUDA_CHECK(cudaMalloc(&dC,bytes));
 CUDA_CHECK(cudaMemcpy(dA,A.data(),bytes,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dB,B.data(),bytes,cudaMemcpyHostToDevice));
 dim3 block(16,16),grid((N+15)/16,(N+15)/16);
 cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e)); CUDA_CHECK(cudaEventRecord(s));
 matMulGlobal<<<grid,block>>>(dA,dB,dC,N); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
 float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,s,e)); CUDA_CHECK(cudaMemcpy(C.data(),dC,bytes,cudaMemcpyDeviceToHost));
 float err=0;for(int i=0;i<N*N;++i)err=std::max(err,std::fabs(C[i]-ref[i]));
 std::cout<<"Global-memory matrix multiplication; N="<<N<<"\nKernel time (ms): "<<ms<<"\nMax absolute error: "<<err<<"\nValidation: "<<(err<1e-2?"PASS":"CHECK")<<"\n";
 CUDA_CHECK(cudaEventDestroy(s));CUDA_CHECK(cudaEventDestroy(e));CUDA_CHECK(cudaFree(dA));CUDA_CHECK(cudaFree(dB));CUDA_CHECK(cudaFree(dC));return err<1e-2?0:1;
}