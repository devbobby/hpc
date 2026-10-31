#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<"\n"; std::exit(EXIT_FAILURE);} } while(0)
constexpr int BINS=16;
__global__ void histogram(const int*in,unsigned int*h,int N){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N){int b=in[i];if(b>=0&&b<BINS)atomicAdd(&h[b],1U);}}
int main(){
 const int N=1<<20;std::vector<int>in(N);std::vector<unsigned int>expected(BINS,0),out(BINS,0);
 for(int i=0;i<N;++i){in[i]=i%BINS;expected[in[i]]++;}
 int *di;unsigned int *dh;CUDA_CHECK(cudaMalloc(&di,N*sizeof(int)));CUDA_CHECK(cudaMalloc(&dh,BINS*sizeof(unsigned int)));
 CUDA_CHECK(cudaMemcpy(di,in.data(),N*sizeof(int),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemset(dh,0,BINS*sizeof(unsigned int)));
 int threads=256,blocks=(N+threads-1)/threads;histogram<<<blocks,threads>>>(di,dh,N);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());CUDA_CHECK(cudaMemcpy(out.data(),dh,BINS*sizeof(unsigned int),cudaMemcpyDeviceToHost));
 bool ok=out==expected;std::cout<<"Histogram using atomicAdd\n";for(int i=0;i<BINS;++i)std::cout<<"bin "<<i<<": "<<out[i]<<"\n";std::cout<<"Validation: "<<(ok?"PASS":"CHECK")<<"\n";
 CUDA_CHECK(cudaFree(di));CUDA_CHECK(cudaFree(dh));return ok?0:1;
}