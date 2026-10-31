Experiment 11: CUDA Performance Optimization and cuBLAS

Files: cpu_matmul.cpp, basic_cuda.cu, tiled_cuda.cu, cublas_matmul.cu

Requirements: NVIDIA CUDA GPU, CUDA Toolkit (nvcc), C++ compiler, cuBLAS. Google Colab GPU runtime is an alternative.

Compile in Linux/Colab terminal (after placing files in current folder):
  g++ -O2 cpu_matmul.cpp -o cpu_matmul
  nvcc -O2 basic_cuda.cu -o basic_cuda
  nvcc -O2 tiled_cuda.cu -o tiled_cuda
  nvcc -O2 cublas_matmul.cu -lcublas -o cublas_matmul

Run for each size 256, 512, 1024:
  ./cpu_matmul 256
  ./basic_cuda 256 8
  ./basic_cuda 256 16
  ./tiled_cuda 256 8
  ./tiled_cuda 256 16
  ./cublas_matmul 256

CUDA event times report kernel/library time and exclude host-device transfers. Run each case more than once and record actual measurements. The included programs print checksums as a basic sanity check; for formal correctness verification, compare every output element with a CPU reference using a tolerance. Do not report illustrative report values as measured results.
