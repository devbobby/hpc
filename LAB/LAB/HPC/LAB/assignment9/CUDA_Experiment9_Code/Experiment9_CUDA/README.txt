EXPERIMENT 9 — CUDA FUNDAMENTALS
=================================

Contents
--------
src/gpu_info.cu                 Query CUDA GPU properties
src/hello_cuda.cu               Hello from each GPU thread
src/thread_ids.cu               Block, local thread and global thread IDs
src/vector_add_cuda.cu          CUDA vector addition + correctness + elapsed time
src/memory_management.cu        Demonstrates cudaMalloc/cudaMemcpy/cudaFree
src/block_size_benchmark.cu     Tests 64, 128, 256 and 512 threads per block
src/compare_cpu_omp_cuda.cu     Sequential vs OpenMP vs CUDA comparison
src/array_squaring.cu           Array squaring on CPU and GPU

IMPORTANT
---------
CUDA programs require an NVIDIA GPU supported by CUDA, an NVIDIA driver,
and the CUDA Toolkit (nvcc). A standard CPU-only computer cannot execute
CUDA kernels. Record actual system information and measurements; do not
copy sample timings as if they were measured on your computer.

UBUNTU: check environment
-------------------------
nvidia-smi
nvcc --version
g++ --version

Compile and run (from the Experiment9_CUDA folder)
--------------------------------------------------
mkdir -p build
nvcc src/gpu_info.cu -o build/gpu_info
./build/gpu_info

nvcc src/hello_cuda.cu -o build/hello_cuda
./build/hello_cuda 1 8
./build/hello_cuda 1 16
./build/hello_cuda 2 8
./build/hello_cuda 4 8

nvcc src/thread_ids.cu -o build/thread_ids
./build/thread_ids

nvcc src/vector_add_cuda.cu -o build/vector_add_cuda
./build/vector_add_cuda 1000000 256
./build/vector_add_cuda 1000000 64

nvcc src/memory_management.cu -o build/memory_management
./build/memory_management

nvcc src/block_size_benchmark.cu -o build/block_size_benchmark
./build/block_size_benchmark 1000000

OpenMP + CUDA comparison (Linux)
--------------------------------
nvcc -Xcompiler -fopenmp src/compare_cpu_omp_cuda.cu -o build/compare_cpu_omp_cuda
./build/compare_cpu_omp_cuda 100000
./build/compare_cpu_omp_cuda 1000000
./build/compare_cpu_omp_cuda 10000000
Set OpenMP thread count if desired:
OMP_NUM_THREADS=4 ./build/compare_cpu_omp_cuda 1000000

Array squaring
--------------
nvcc src/array_squaring.cu -o build/array_squaring
./build/array_squaring

Windows (Developer PowerShell / terminal with CUDA Toolkit)
------------------------------------------------------------
Use nvcc to compile, e.g.:
nvcc src\\gpu_info.cu -o build\\gpu_info.exe
build\\gpu_info.exe
For OpenMP comparison, use a supported Visual Studio host compiler and:
nvcc -Xcompiler /openmp src\\compare_cpu_omp_cuda.cu -o build\\compare_cpu_omp_cuda.exe
Windows compiler/toolkit combinations can differ; if this command fails, check
that Visual Studio C++ build tools and a compatible CUDA Toolkit are installed.

Concepts and formulas
--------------------
Global thread ID (1-D grid):
    id = blockIdx.x * blockDim.x + threadIdx.x

Blocks required:
    blocks = (N + threadsPerBlock - 1) / threadsPerBlock

Speedup:
    speedup = sequential_time / parallel_time

Boundary check:
    if (id < N) { ... }

Memory movement:
    Host arrays --cudaMemcpyHostToDevice--> Device arrays
    Kernel operates on device arrays
    Device result --cudaMemcpyDeviceToHost--> Host result

Measurement note
----------------
block_size_benchmark reports kernel-only time using CUDA events after warm-up.
compare_cpu_omp_cuda reports CPU wall time and CUDA total wall time including
memory transfers, so these measurements are not the same metric. State which
metric is being compared in the report. Run each experiment more than once
and use a consistent method for the final table.
