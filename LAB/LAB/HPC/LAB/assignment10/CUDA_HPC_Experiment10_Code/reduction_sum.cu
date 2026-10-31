#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cstdlib>
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<"\n"; std::exit(EXIT_FAILURE);} } while(0)
__global__ void reduceSum(const float*in,float*partial,int N){
 extern __shared__ float s[]; unsigned tid=threadIdx.x; unsigned i=blockIdx.x*(blockDim.x*2)+tid;
 float v=0;if(i<N)v+=in[i];if(i+blockDim.x<N)v+=in[i+blockDim.x];s[tid]=v;__syncthreads();
 for(unsigned stride=blockDim.x/2;stride>0;stride>>=1){if(tid<stride)s[tid]+=s[tid+stride];__syncthreads();}
 if(tid==0)partial[blockIdx.x]=s[0];
}
int main(){
 const int N=1<<20,threads=256,blocks=(N+threads*2-1)/(threads*2);std::vector<float>in(N,1.0f);
 float *di,*dp;CUDA_CHECK(cudaMalloc(&di,N*sizeof(float)));CUDA_CHECK(cudaMalloc(&dp,blocks*sizeof(float)));CUDA_CHECK(cudaMemcpy(di,in.data(),N*sizeof(float),cudaMemcpyHostToDevice));
 reduceSum<<<blocks,threads,threads*sizeof(float)>>>(di,dp,N);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
 std::vector<float>p(blocks);CUDA_CHECK(cudaMemcpy(p.data(),dp,blocks*sizeof(float),cudaMemcpyDeviceToHost));float result=std::accumulate(p.begin(),p.end(),0.0f);
 std::cout<<"Parallel sum reduction\nN="<<N<<"\nResult="<<result<<"\nExpected="<<N<<"\nValidation: "<<(result==N?"PASS":"CHECK")<<"\n";
 CUDA_CHECK(cudaFree(di));CUDA_CHECK(cudaFree(dp));return result==N?0:1;
}