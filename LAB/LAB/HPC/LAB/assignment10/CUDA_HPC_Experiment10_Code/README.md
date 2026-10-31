# CUDA / HPC Experiment 10 — Sample Code

Contains CUDA C++ examples:
1. Basic matrix multiplication using global memory
2. Tiled matrix multiplication using shared memory
3. Parallel sum reduction
4. Inclusive prefix sum (single-block educational implementation)
5. Histogram using atomic operations
6. Performance measurement notes

## Requirements
- NVIDIA GPU that supports CUDA
- NVIDIA CUDA Toolkit (`nvcc`)
- Compatible NVIDIA driver

CUDA will not execute on an Intel integrated GPU alone. Use a university lab machine or a cloud instance with CUDA support if your laptop lacks a compatible NVIDIA GPU.

## Compile and run (Linux / Ubuntu)
Run from this folder:
```bash
nvcc -O2 matrix_global.cu -o matrix_global && ./matrix_global
nvcc -O2 matrix_tiled.cu -o matrix_tiled && ./matrix_tiled
nvcc -O2 reduction_sum.cu -o reduction_sum && ./reduction_sum
nvcc -O2 prefix_sum.cu -o prefix_sum && ./prefix_sum
nvcc -O2 histogram.cu -o histogram && ./histogram
```

On Windows Developer Command Prompt, compile with the same `nvcc` commands and run the generated `.exe` files.

## Notes
- These are educational examples, not a guarantee of the exact assignment rubric.
- Prefix sum is a single-block example and uses at most 1024 elements.
- Timing varies by GPU, driver, CUDA version, and system load.
