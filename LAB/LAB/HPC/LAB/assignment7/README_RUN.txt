# Experiment 7 - MPI Collective Communication and Parallel Matrix Computations

## Compile
mpicc 01_bcast.c -o bcast
mpicc 02_scatter_gather.c -o scatter_gather
mpicc 03_reduce.c -o reduce
mpicc 04_matvec.c -o matvec
mpicc 05_matmat.c -o matmat
mpicc 06_decomposition_compare.c -o decomposition_compare -lm
mpicc 06b_block_matmat.c -o block_matmat -lm
mpicc 07_scatterv_matvec.c -o scatterv_matvec

## Run examples
mpirun -np 4 ./bcast
mpirun -np 4 ./scatter_gather
mpirun -np 4 ./reduce

mpirun -np 2 ./matvec 500
mpirun -np 4 ./matvec 500
mpirun -np 8 ./matvec 500
mpirun -np 2 ./matvec 1000
mpirun -np 4 ./matvec 1000
mpirun -np 8 ./matvec 1000
mpirun -np 2 ./matvec 2000
mpirun -np 4 ./matvec 2000
mpirun -np 8 ./matvec 2000

mpirun -np 2 ./matmat 500
mpirun -np 4 ./matmat 500
mpirun -np 8 ./matmat 500
mpirun -np 2 ./matmat 1000
mpirun -np 4 ./matmat 1000
mpirun -np 8 ./matmat 1000

mpirun -np 4 ./decomposition_compare 400
mpirun -np 4 ./block_matmat 400

mpirun -np 3 ./scatterv_matvec 10
