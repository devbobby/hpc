# Experiment 8 — Comparative Study of OpenMP and MPI

This package implements all six source-code deliverables listed in the assignment:
1. Sequential matrix multiplication
2. OpenMP matrix multiplication
3. MPI matrix multiplication
4. Sequential 2D convolution
5. OpenMP 2D convolution
6. MPI 2D convolution

It also includes the theory, setup instructions, experiment procedure, formulas, report outline, and discussion guidance.

## 1. Theory and concepts

### OpenMP
OpenMP is a shared-memory parallel programming API for C/C++ and Fortran. A program creates a team of threads that can access the same process memory. `#pragma omp parallel for` divides loop iterations among threads. It is typically straightforward on a multicore computer and usually needs no explicit message passing for shared arrays. Threads still need synchronization when updating shared data.

### MPI
MPI (Message Passing Interface) is a library/API for processes that may have separate address spaces and may run on separate machines. Processes communicate explicitly through messages. This package uses `MPI_Scatterv` to distribute rows, `MPI_Bcast` to broadcast matrix B, `MPI_Gatherv` to collect results, and `MPI_Sendrecv` for convolution halo exchange.

### Thread vs process
- Thread: execution path inside a process; threads can share the process's memory.
- MPI process: independent process with its own address space; data is exchanged by messages.

### Matrix multiplication
For matrices A and B of size N×N, calculate `C[i][j] = sum(A[i][k] * B[k][j])` over k. Each output row can be calculated independently, so OpenMP distributes rows among threads. The MPI version distributes row blocks of A, broadcasts B to each process, computes local C rows, and gathers the result.

### 2D convolution
Each output cell combines a cell's neighborhood with a 3×3 kernel. The included kernel is all ones. Output boundary cells are left at zero. OpenMP parallelizes independent output rows/cells. MPI partitions input rows and exchanges neighboring boundary rows, called halo/ghost rows, so each process can calculate outputs near its partition boundary.

### Performance metrics
Let `T1` be sequential runtime and `Tp` be runtime with p threads/processes:
- Speedup: `S(p) = T1 / Tp`
- Efficiency: `E(p) = S(p) / p` (multiply by 100 for percent)
- Strong scaling: fixed total problem size, increase workers.
- Weak scaling: increase problem size in proportion to worker count.

Timing differences depend on CPU, memory bandwidth, compiler, process launch overhead, workload size, and system load. Do not invent or copy sample timings; record your own measured values.

## 2. Requirements

- A C++ compiler such as `g++`
- OpenMP support in the compiler
- An MPI implementation, such as Open MPI or MPICH (`mpic++` and `mpirun`/`mpiexec`)
- Terminal / command prompt

### Ubuntu / Debian Linux
```bash
sudo apt update
sudo apt install g++ make openmpi-bin libopenmpi-dev
```

### Windows
Recommended: install Ubuntu using WSL2, then follow the Linux commands above. Native Windows MPI can also work if MPI and its compiler wrapper are correctly installed.

### macOS
Install a compiler and MPI using Homebrew (for example, `brew install gcc open-mpi`). Exact compiler executable names may vary.

## 3. Compile

Open a terminal in this extracted folder.

```bash
g++ -O2 -std=c++17 matrix_seq.cpp -o matrix_seq
g++ -O2 -std=c++17 -fopenmp matrix_openmp.cpp -o matrix_omp
mpic++ -O2 -std=c++17 matrix_mpi.cpp -o matrix_mpi

g++ -O2 -std=c++17 convolution_seq.cpp -o convolution_seq
g++ -O2 -std=c++17 -fopenmp convolution_openmp.cpp -o convolution_omp
mpic++ -O2 -std=c++17 convolution_mpi.cpp -o convolution_mpi
```

If `mpic++` is not found, MPI is either not installed or its executable directory is not on PATH. On some systems the wrapper is named `mpicxx`.

## 4. Run examples

The first argument is problem size N. For OpenMP, the second argument is thread count. For OpenMP convolution, the third argument is scheduling strategy.

```bash
# Matrix multiplication
./matrix_seq 500
./matrix_omp 500 2
./matrix_omp 500 4
./matrix_omp 500 8
mpirun -np 2 ./matrix_mpi 500
mpirun -np 4 ./matrix_mpi 500
mpirun -np 8 ./matrix_mpi 500

# 2D convolution
./convolution_seq 1000
./convolution_omp 1000 2 static
./convolution_omp 1000 4 dynamic
./convolution_omp 1000 8 guided
mpirun -np 2 ./convolution_mpi 1000
mpirun -np 4 ./convolution_mpi 1000
mpirun -np 8 ./convolution_mpi 1000
```

Try assignment sizes too:
- Matrix multiplication: 500, 1000, 2000
- Convolution: 1000, 2000, 4000

Large inputs need substantial RAM and runtime. Start small to verify compilation. For a laptop with fewer than 8 cores, running 8 threads/processes is still possible in many configurations but may oversubscribe the available CPU and become slower. Some MPI installations refuse to run as root; do not bypass that restriction unless you understand the security implications.

## 5. Suggested experiment workflow

1. Extract the ZIP.
2. Install compiler/OpenMP/MPI.
3. Compile all six programs using the commands above.
4. Run every program with the same N for a fair comparison.
5. Record the printed `Time_s` and `Checksum` values. A matching checksum across sequential/OpenMP/MPI versions of the same application is a useful correctness check (small floating-point differences can occur in other implementations).
6. Repeat each test at least three times and use the median time, if practical. Avoid running other heavy tasks during timing.
7. Test OpenMP with 2, 4, 8 and maximum available threads; test MPI with 2, 4 and 8 processes.
8. Calculate speedup and efficiency from the sequential runtime and create the required graphs.
9. Explain results based on measurements from your machine, not expected or fabricated values.

## 6. Result table template

| Application | N | Model | Threads/processes | Time (s) | Speedup | Efficiency (%) |
|---|---:|---|---:|---:|---:|---:|
| Matrix multiplication | 500 | Sequential | 1 | | 1.00 | 100 |
| Matrix multiplication | 500 | OpenMP | 2 | | | |
| Matrix multiplication | 500 | OpenMP | 4 | | | |
| Matrix multiplication | 500 | OpenMP | 8 | | | |
| Matrix multiplication | 500 | MPI | 2 | | | |
| 2D convolution | 1000 | Sequential | 1 | | 1.00 | 100 |
| 2D convolution | 1000 | OpenMP | 2 | | | |
| 2D convolution | 1000 | MPI | 2 | | | |

For each parallel row, calculate speedup as `T_sequential / T_parallel`, and efficiency as `speedup / workers × 100%`.

## 7. How to interpret the results

- Small problems may not benefit from parallelism because thread startup, MPI communication, and process-management costs can be comparable to the computation.
- OpenMP often has lower communication overhead on a single shared-memory node because threads access shared arrays directly.
- MPI may be slower for small workloads because distributing inputs, broadcasting data, halo exchange, and gathering results take time.
- MPI can scale across nodes because processes communicate through messages rather than requiring one shared address space.
- More workers do not guarantee better performance. Memory bandwidth, cache behavior, load imbalance, synchronization, and communication can limit scaling.
- Matrix multiplication performs many arithmetic operations and is a useful parallel workload. This simple implementation is educational, not a highly optimized BLAS implementation.
- Convolution uses local neighborhoods. MPI must exchange halo rows to calculate values adjacent to process boundaries correctly.

## 8. Architecture comparison

| Feature | OpenMP | MPI |
|---|---|---|
| Memory model | Shared memory | Distributed address spaces |
| Execution unit | Thread | Process/rank |
| Communication | Shared variables/arrays | Explicit messages/collectives |
| Synchronization | Barriers, critical/atomic, reductions | Message ordering and collective operations |
| Data distribution | Shared data; loop work divided among threads | Programmer distributes data among ranks |
| Overhead | Often lower within one node | Message and collective communication overhead |
| Scalability | Primarily within a shared-memory node | Across nodes and clusters |
| Complexity | Often simpler for loop parallelism | More explicit data/communication design |

## 9. Discussion / theory answers

1. **Why can one algorithm use both models?** Independent work can be assigned to threads sharing data or to processes holding separate portions of data; the mathematical computation does not dictate one programming model.
2. **Fundamental difference?** OpenMP uses threads in a shared address space; MPI uses separate processes and explicit messages.
3. **Which requires explicit communication?** MPI.
4. **Why distribute data in MPI?** Each process has its own memory, so required input must be sent or independently created on each rank.
5. **Why lower OpenMP communication overhead on one node?** Threads can read/write shared arrays without sending messages, though synchronization and memory contention still cost time.
6. **Why can MPI scale across nodes?** Ranks communicate over network interconnects and do not require a single shared-memory system.
7. **Why is matrix multiplication suitable for both?** Many output rows/cells can be computed independently, giving abundant parallel work.
8. **Why convolution boundaries need care?** Outputs near a partition boundary depend on input rows owned by a neighboring rank.
9. **What is a halo/ghost region?** Extra copies of neighboring data stored locally so stencil calculations can access adjacent cells.
10. **Why can scaling stop?** Serial work, communication, synchronization, memory bandwidth, load imbalance, and worker-management costs limit speedup.
11. **How does problem size affect speedup?** Larger workloads can amortize parallel overhead; very large problems can also hit memory or communication limits.
12. **How select OpenMP or MPI?** Consider hardware topology, data size, communication pattern, implementation complexity, memory capacity, and scaling requirements.

## 10. Hybrid MPI + OpenMP challenge

For 4 nodes with 8 cores per node, a conceptual design is 1 MPI rank per node and 8 OpenMP threads per rank. MPI distributes blocks of A and broadcasts or distributes required portions of B between ranks; OpenMP threads compute assigned local output rows. A hybrid design can reduce the number of MPI ranks and use shared memory within each node. Challenges include thread affinity, load balancing, communication between nodes, and avoiding oversubscription. This is a conceptual design; this package does not include a hybrid executable.

## 11. Suggested 300–500 word AI/HPC case study: image processing

Large-scale image processing often applies the same local or global operation to millions of pixels. Examples include smoothing, sharpening, edge detection, medical-image enhancement, and preprocessing images for computer-vision models. These operations can be computationally intensive when image dimensions, image counts, or processing stages increase.

OpenMP is suitable when one computer has multiple CPU cores and the image fits in shared memory. The image can be stored in a shared array, and threads can process separate rows or tiles. In a 2D convolution, each output pixel depends on a small neighborhood of input pixels. Since most output pixels can be computed independently, a parallel loop is a natural fit. The implementation must avoid multiple threads writing the same output cell and must preserve correct boundary behavior.

MPI is useful when the image is too large for one machine, or when a large collection of images is distributed over a cluster. The image can be split into row blocks or tiles, and each MPI rank computes its assigned region. For convolution, ranks exchange halo rows or border pixels with neighboring ranks because stencil operations require nearby input values. This communication adds overhead, so each rank should ideally perform enough computation to make the exchange worthwhile.

Scalability depends on the image size, kernel size, memory bandwidth, number of workers, and communication cost. Increasing the worker count can reduce computation time, but speedup is not always linear. Small images may finish before parallel overhead is recovered, while large images may become limited by memory traffic or network communication.

A hybrid MPI + OpenMP approach can combine both models. MPI can distribute image regions across compute nodes, while OpenMP threads process each region inside a node. This can reduce the number of MPI ranks and use shared memory efficiently. A practical design must balance tile sizes, halo-exchange frequency, thread placement, and memory usage. The most appropriate model should be selected by measuring execution time, speedup, efficiency, and correctness on the target system.

## 12. Report outline

1. Aim and problem statement
2. Theory: OpenMP, MPI, shared/distributed memory
3. Hardware/software setup (CPU, cores, RAM, OS, compiler, MPI version)
4. Algorithms and source-code explanation
5. Experimental method and input sizes
6. Raw timing tables and checksum/correctness checks
7. Speedup and efficiency calculations
8. Required graphs
9. Architecture and communication analysis
10. AI/HPC case study
11. Answers to reflection questions based on your measured results
12. Conclusion

**Important:** timings depend on your computer. Run the experiments and fill the result tables with your own data; the package intentionally does not fabricate benchmark results.
