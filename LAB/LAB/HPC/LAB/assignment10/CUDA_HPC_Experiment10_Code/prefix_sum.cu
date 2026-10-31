#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<"\n"; std::exit(EXIT_FAILURE);} } while(0)
__global__ void inclusiveScan(const int*in,int*out,int N){
 __shared__ int temp[1024];int tid=threadIdx.x;if(tid<N)temp[tid]=in[tid];__syncthreads();
 for(int offset=1;offset<N;offset<<=1){
  int add=(tid<N&&tid>=offset)?temp[tid-offset]:0;__syncthreads();
  if(tid<N)temp[tid]+=add;__syncthreads();
 }
 if(tid<N)out[tid]=temp[tid];
}
int main(){
 const int N=16;std::vector<int>in(N),out(N);for(int i=0;i<N;++i)in[i]=i+1;
 int *di,*dout;CUDA_CHECK(cudaMalloc(&di,N*sizeof(int)));CUDA_CHECK(cudaMalloc(&dout,N*sizeof(int)));CUDA_CHECK(cudaMemcpy(di,in.data(),N*sizeof(int),cudaMemcpyHostToDevice));
 inclusiveScan<<<1,1024>>>(di,dout,N);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());CUDA_CHECK(cudaMemcpy(out.data(),dout,N*sizeof(int),cudaMemcpyDeviceToHost));
 int sum=0;bool ok=true;std::cout<<"Input: ";for(int x:in)std::cout<<x<<" ";std::cout<<"\nInclusive scan: ";
 for(int i=0;i<N;++i){sum+=in[i];std::cout<<out[i]<<" ";if(out[i]!=sum)ok=false;}std::cout<<"\nValidation: "<<(ok?"PASS":"CHECK")<<"\n";
 CUDA_CHECK(cudaFree(di));CUDA_CHECK(cudaFree(dout));return ok?0:1;
}