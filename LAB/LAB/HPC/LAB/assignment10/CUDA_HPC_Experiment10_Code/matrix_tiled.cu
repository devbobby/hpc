#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<"\n"; std::exit(EXIT_FAILURE);} } while(0)
constexpr int TILE=16;
__global__ void matMulTiled(const float*A,const float*B,float*C,int N){
 __shared__ float a[TILE][TILE],b[TILE][TILE];
 int row=blockIdx.y*TILE+threadIdx.y,col=blockIdx.x*TILE+threadIdx.x; float sum=0;
 int nt=(N+TILE-1)/TILE;
 for(int t=0;t<nt;++t){
  int ac=t*TILE+threadIdx.x,br=t*TILE+threadIdx.y;
  a[threadIdx.y][threadIdx.x]=(row<N&&ac<N)?A[row*N+ac]:0;
  b[threadIdx.y][threadIdx.x]=(br<N&&col<N)?B[br*N+col]:0;
  __syncthreads();
  for(int k=0;k<TILE;++k)sum+=a[threadIdx.y][k]*b[k][threadIdx.x];
  __syncthreads();
 }
 if(row<N&&col<N)C[row*N+col]=sum;
}
int main(){
 const int N=128;size_t bytes=(size_t)N*N*sizeof(float);std::vector<float>A(N*N),B(N*N),C(N*N),ref(N*N,0);
 for(int i=0;i<N*N;++i){A[i]=(i%13+1)/13.0f;B[i]=(i%7+1)/7.0f;}
 for(int r=0;r<N;++r)for(int c=0;c<N;++c)for(int k=0;k<N;++k)ref[r*N+c]+=A[r*N+k]*B[k*N+c];
 float *dA,*dB,*dC;CUDA_CHECK(cudaMalloc(&dA,bytes));CUDA_CHECK(cudaMalloc(&dB,bytes));CUDA_CHECK(cudaMalloc(&dC,bytes));
 CUDA_CHECK(cudaMemcpy(dA,A.data(),bytes,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dB,B.data(),bytes,cudaMemcpyHostToDevice));
 dim3 block(TILE,TILE),grid((N+TILE-1)/TILE,(N+TILE-1)/TILE);
 cudaEvent_t s,e;CUDA_CHECK(cudaEventCreate(&s));CUDA_CHECK(cudaEventCreate(&e));CUDA_CHECK(cudaEventRecord(s));
 matMulTiled<<<grid,block>>>(dA,dB,dC,N);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaEventRecord(e));CUDA_CHECK(cudaEventSynchronize(e));
 float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,s,e));CUDA_CHECK(cudaMemcpy(C.data(),dC,bytes,cudaMemcpyDeviceToHost));
 float err=0;for(int i=0;i<N*N;++i)err=std::max(err,std::fabs(C[i]-ref[i]));
 std::cout<<"Shared-memory tiled matrix multiplication; N="<<N<<"\nKernel time (ms): "<<ms<<"\nMax absolute error: "<<err<<"\nValidation: "<<(err<1e-2?"PASS":"CHECK")<<"\n";
 CUDA_CHECK(cudaEventDestroy(s));CUDA_CHECK(cudaEventDestroy(e));CUDA_CHECK(cudaFree(dA));CUDA_CHECK(cudaFree(dB));CUDA_CHECK(cudaFree(dC));return err<1e-2?0:1;
}