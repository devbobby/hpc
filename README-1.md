MPI

MESSAGE PASSING INTERFACE

Complete HPC Practical & Exam Notes

Concepts • Installation • Processes • Ranks • Groups • Communicators
Point-to-point • Collective communication • Matrix addition • Dot
product

Simple explanations + complete C programs + compile/run commands + viva
questions

# Contents

-   1.  MPI overview and distributed-memory concept
-   2.  Installing and verifying MPI
-   3.  MPI program structure and execution
-   4.  Processes, ranks, size, groups and communicators
-   5.  Point-to-point communication: MPI_Send and MPI_Recv
-   6.  Non-blocking communication: MPI_Isend and MPI_Irecv
-   7.  Collective communication overview
-   8.  MPI_Bcast - broadcast
-   9.  MPI_Scatter - scatter
-   10. MPI_Gather - gather
-   11. MPI_Reduce and MPI_Allreduce - reduction
-   12. MPI_Barrier - synchronization
-   13. Sequential matrix addition (reference program)
-   14. Parallel matrix addition using MPI
-   15. Dot product using MPI
-   16. Important differences and common mistakes
-   17. Viva questions and quick revision

Scope: This document consolidates the MPI topics discussed in the
conversation. It intentionally leaves out OpenMP and CUDA material.

## 1. MPI overview and distributed-memory concept

MPI stands for Message Passing Interface. It is a standard programming
interface used to develop parallel programs in which multiple processes
communicate by sending and receiving messages.

In MPI, each process has its own private memory. One process cannot
directly read or write another process's memory. When data must move
between processes, MPI communication functions are used.

| Term \| Meaning \|

| --- \| --- \|

| Parallel computing \| Solving a problem by dividing work among
  multiple processing units. \|

| Process \| An independent running instance of an MPI program. \|

| Message passing \| Sending data from one process to another. \|

| Distributed memory \| Each process has its own address space / memory.
  \|

| MPI \| A standard API that defines communication routines; it is not
  itself a programming language. \|

### Why MPI is used

-   To run programs on multiple CPU cores or on multiple computers in a
    cluster.

-   To divide a large problem into smaller parts and solve them in
    parallel.

-   To exchange partial data and combine partial results.

-   To write portable parallel programs across systems that support an
    MPI implementation.

### Simple example

Suppose there are 4 processes and an array contains 1000 values. The
work can be divided so that each process handles 250 values. Each
process calculates a partial answer, and MPI can combine the partial
answers.

## 2. Installing and verifying MPI

A common MPI implementation for lab work is Open MPI. Install the
compiler wrapper and development headers so that C programs can include
mpi.h.

### Ubuntu / Debian

sudo apt update

sudo apt install openmpi-bin libopenmpi-dev

### Fedora

sudo dnf install openmpi openmpi-devel

Fedora note: Depending on the Fedora package setup, MPI binaries may be
installed in an Open MPI-specific directory or exposed through
environment modules. Check the package instructions if mpicc is not
found.

### Verify installation

mpicc --version

mpirun --version

which mpicc

If these commands show compiler / Open MPI information and the mpicc
path, the installation is generally ready. mpicc is a compiler wrapper
that invokes the C compiler with the required MPI include paths and
libraries.

## 3. MPI program structure and execution

The usual order of an MPI program is: initialize MPI, find each process
rank and the total process count, perform communication and computation,
then finalize MPI.

File: hello_mpi.c

/\* Compile: mpicc hello_mpi.c -o hello_mpi

Run: mpirun -np 4 ./hello_mpi \*/

``` c
#include <stdio.h>
```

``` c
#include <mpi.h>
```

int main(int argc, char \*argv\[\]) {

int rank, size;

MPI_Init(&argc, &argv);

MPI_Comm_rank(MPI_COMM_WORLD, &rank);

MPI_Comm_size(MPI_COMM_WORLD, &size);

printf("Hello from process %d of %d`\n`{=tex}", rank, size);

MPI_Finalize();

return 0;

}

Compile: mpicc hello_mpi.c -o hello_mpi

Run with 4 processes: mpirun -np 4 ./hello_mpi

Expected idea: four processes print a hello message with ranks 0, 1, 2
and 3. The order of the printed lines can change between runs because
processes execute concurrently.

| Function \| Purpose \|

| --- \| --- \|

| MPI_Init(&argc, &argv) \| Starts the MPI execution environment. \|

| MPI_Comm_rank(comm, &rank) \| Gets the calling process rank in a
  communicator. \|

| MPI_Comm_size(comm, &size) \| Gets the number of processes in a
  communicator. \|

| MPI_Finalize() \| Shuts down the MPI environment; call it before
  normal program exit. \|

Remember: Every MPI process runs the same program, but rank-based if
conditions can make different processes do different work.

## 4. Processes, ranks, groups and communicators

### 4.1 Process and rank

A process is one running copy of the MPI program. Rank is the unique
number assigned to a process inside a communicator. Ranks normally start
from 0.

For a program started with mpirun -np 4, MPI_COMM_WORLD contains 4
processes: rank 0, rank 1, rank 2 and rank 3.

### 4.2 Communicator

A communicator defines a communication context and the set of processes
allowed to communicate with one another. MPI_COMM_WORLD is the default
communicator containing all processes launched for the MPI program.

### 4.3 Group

An MPI group is an ordered collection of process identifiers (ranks). A
group describes membership and rank ordering. A group alone is not the
communication context used for message exchange; a communicator combines
a group with a communication context.

| Concept \| What it represents \| Example \|

| --- \| --- \| --- \|

| Rank \| ID of one process in a communicator \| Rank 2 \|

| Size \| Number of processes in a communicator \| Size = 4 \|

| Group \| Ordered set of processes \| Processes with world ranks 0 and
  1 \|

| Communicator \| Group plus a separate communication context \|
  MPI_COMM_WORLD \|

### 4.4 Creating subcommunicators with MPI_Comm_split

MPI_Comm_split divides the processes of an existing communicator into
smaller communicators. In this example, even ranks form one communicator
and odd ranks form another.

File: split_groups.c

/\* Compile: mpicc split_groups.c -o split_groups

Run: mpirun -np 4 ./split_groups \*/

``` c
#include <stdio.h>
```

``` c
#include <mpi.h>
```

int main(int argc, char \*argv\[\]) {

int rank, subrank, color;

MPI_Comm subcomm;

MPI_Init(&argc, &argv);

MPI_Comm_rank(MPI_COMM_WORLD, &rank);

color = rank % 2; // 0: even ranks, 1: odd ranks

MPI_Comm_split(MPI_COMM_WORLD, color, rank, &subcomm);

MPI_Comm_rank(subcomm, &subrank);

printf("World rank %d belongs to group %d; sub-rank = %d`\n`{=tex}",

rank, color, subrank);

MPI_Comm_free(&subcomm);

MPI_Finalize();

return 0;

}

Compile / run: mpicc split_groups.c -o split_groups \| mpirun -np 4
./split_groups

With 4 processes, world ranks 0 and 2 are in color 0; ranks 1 and 3 are
in color 1. Their sub-ranks start from 0 inside their own
subcommunicator. Output order may vary.

## 5. Point-to-point communication: MPI_Send and MPI_Recv

Point-to-point communication means one process sends a message directly
to another process. The two most common blocking functions are MPI_Send
and MPI_Recv.

### 5.1 MPI_Send

MPI_Send sends data from the calling process to a destination process.

Syntax: MPI_Send(buffer, count, datatype, destination, tag,
communicator);

### 5.2 MPI_Recv

MPI_Recv receives data from a source process. It waits until the
matching message is received.

Syntax: MPI_Recv(buffer, count, datatype, source, tag, communicator,
status);

| Argument \| Meaning \|

| --- \| --- \|

| buffer \| Memory address of the data to send or receive. \|

| count \| Maximum number of data elements. \|

| datatype \| MPI_INT, MPI_FLOAT, MPI_DOUBLE, MPI_CHAR, etc. \|

| destination / source \| Rank of the receiving / sending process. \|

| tag \| Integer message label used to match messages. \|

| communicator \| Communication context, commonly MPI_COMM_WORLD. \|

| status \| Information about the received message; MPI_STATUS_IGNORE
  can be used when details are not needed. \|

### Complete example: process 0 sends a number to process 1

File: send_recv.c

/\* Compile: mpicc send_recv.c -o send_recv

Run: mpirun -np 2 ./send_recv \*/

``` c
#include <stdio.h>
```

``` c
#include <mpi.h>
```

int main(int argc, char \*argv\[\]) {

int rank, size, number;

MPI_Init(&argc, &argv);

MPI_Comm_rank(MPI_COMM_WORLD, &rank);

MPI_Comm_size(MPI_COMM_WORLD, &size);

if (size \< 2) {

if (rank == 0) printf("Run with at least 2 processes.`\n`{=tex}");

} else if (rank == 0) {

number = 100;

MPI_Send(&number, 1, MPI_INT, 1, 10, MPI_COMM_WORLD);

printf("Process 0 sent %d to process 1`\n`{=tex}", number);

} else if (rank == 1) {

MPI_Recv(&number, 1, MPI_INT, 0, 10,

MPI_COMM_WORLD, MPI_STATUS_IGNORE);

printf("Process 1 received %d from process 0`\n`{=tex}", number);

}

MPI_Finalize();

return 0;

}

Compile / run: mpicc send_recv.c -o send_recv \| mpirun -np 2
./send_recv

Expected messages: process 0 sends 100; process 1 receives 100. Which
line appears first is not guaranteed.

Important: The sender and receiver must agree on the communicator,
source/destination, tag and compatible datatype/count. Incorrect
matching can cause the program to wait indefinitely.

## 6. Non-blocking communication: MPI_Isend and MPI_Irecv

Non-blocking communication starts a communication operation and returns
without waiting for the entire operation to finish. It can allow useful
computation to overlap with communication.

| Function \| Purpose \|

| --- \| --- \|

| MPI_Isend(...) \| Starts a non-blocking send. \|

| MPI_Irecv(...) \| Starts a non-blocking receive. \|

| MPI_Wait(&request, &status) \| Waits until the requested operation
  completes. \|

| MPI_Waitall(count, requests, statuses) \| Waits for multiple
  outstanding operations. \|

Important rule: Do not reuse or modify a non-blocking send buffer, and
do not read a non-blocking receive buffer, until the corresponding
request has completed.

For most introductory practicals, first become comfortable with blocking
MPI_Send and MPI_Recv. Non-blocking calls are useful when you want to
overlap computation and communication.

## 7. Collective communication overview

A collective operation involves all processes in a communicator. All
participating processes must call the collective operation in a
compatible order. The function may distribute data, collect data,
broadcast data, combine values or synchronize processes.

| Function \| Meaning \| Simple memory aid \|

| --- \| --- \| --- \|

| MPI_Bcast \| One root sends the same data to all \| Broadcast = same
  data \|

| MPI_Scatter \| Root distributes different portions \| Scatter = divide
  and distribute \|

| MPI_Gather \| Root collects portions from all \| Gather = collect \|

| MPI_Reduce \| Combine values into one root result \| Reduce = combine
  \|

| MPI_Allreduce \| Combine values and give result to all \| Allreduce =
  combine for everyone \|

| MPI_Barrier \| All processes wait at a synchronization point \|
  Barrier = wait \|

Critical exam point: Do not call a collective routine only on the root.
Every process in the communicator must participate, even when only the
root supplies the input or uses the final result.

## 8. MPI_Bcast() - Broadcast

Purpose: Send the same data from one root process to every process in
the communicator. The root also participates in the operation.

Syntax: MPI_Bcast(buffer, count, datatype, root, communicator);

File: bcast.c

/\* Compile: mpicc bcast.c -o bcast

Run: mpirun -np 4 ./bcast \*/

``` c
#include <stdio.h>
```

``` c
#include <mpi.h>
```

int main(int argc, char \*argv\[\]) {

int rank, number = 0;

MPI_Init(&argc, &argv);

MPI_Comm_rank(MPI_COMM_WORLD, &rank);

if (rank == 0) {

printf("Enter a number:");

fflush(stdout);

scanf("%d", &number);

}

MPI_Bcast(&number, 1, MPI_INT, 0, MPI_COMM_WORLD);

printf("Process %d received %d`\n`{=tex}", rank, number);

MPI_Finalize();

return 0;

}

Compile / run: mpicc bcast.c -o bcast \| mpirun -np 4 ./bcast

Example: if rank 0 enters 25, every process prints that it received 25.
Line order can vary.

Parameter meaning: &number is the data buffer; 1 is the number of items;
MPI_INT is the datatype; 0 is the root rank; MPI_COMM_WORLD is the
communicator.

## 9. MPI_Scatter() - Scatter

Purpose: Divide data at the root into equal-sized portions and send one
portion to each process.

Syntax: MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount,
recvtype, root, communicator);

File: scatter.c

/\* Compile: mpicc scatter.c -o scatter

Run: mpirun -np 4 ./scatter \*/

``` c
#include <stdio.h>
```

``` c
#include <mpi.h>
```

int main(int argc, char \*argv\[\]) {

int rank, size;

int data\[8\];

int local\[2\];

MPI_Init(&argc, &argv);

MPI_Comm_rank(MPI_COMM_WORLD, &rank);

MPI_Comm_size(MPI_COMM_WORLD, &size);

if (size != 4) {

if (rank == 0) printf("Run with exactly 4 processes.`\n`{=tex}");

MPI_Finalize();

return 0;

}

if (rank == 0) {

for (int i = 0; i \< 8; i++) {

printf("Enter data\[%d\]:", i);

scanf("%d", &data\[i\]);

}

}

MPI_Scatter(data, 2, MPI_INT,

local, 2, MPI_INT,

0, MPI_COMM_WORLD);

printf("Process %d received: %d %d`\n`{=tex}",

rank, local\[0\], local\[1\]);

MPI_Finalize();

return 0;

}

Compile / run: mpicc scatter.c -o scatter \| mpirun -np 4 ./scatter

For input values 10 20 30 40 50 60 70 80: rank 0 receives 10,20; rank 1
receives 30,40; rank 2 receives 50,60; rank 3 receives 70,80.

The sendcount is the number of elements sent to each process, not the
total number of array elements.

## 10. MPI_Gather() - Gather

Purpose: Collect equal-sized contributions from all processes and place
them in the root process receive array.

Syntax: MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
recvtype, root, communicator);

File: gather.c

/\* Compile: mpicc gather.c -o gather

Run: mpirun -np 4 ./gather \*/

``` c
#include <stdio.h>
```

``` c
#include <mpi.h>
```

int main(int argc, char \*argv\[\]) {

int rank, size;

int gathered\[100\];

int value;

MPI_Init(&argc, &argv);

MPI_Comm_rank(MPI_COMM_WORLD, &rank);

MPI_Comm_size(MPI_COMM_WORLD, &size);

value = rank \* 10;

MPI_Gather(&value, 1, MPI_INT,

gathered, 1, MPI_INT,

0, MPI_COMM_WORLD);

if (rank == 0) {

printf("Gathered values:");

for (int i = 0; i \< size; i++)

printf("%d", gathered\[i\]);

printf("`\n`{=tex}");

}

MPI_Finalize();

return 0;

}

Compile / run: mpicc gather.c -o gather \| mpirun -np 4 ./gather

With 4 processes, local values are 0, 10, 20 and 30. Rank 0 prints:
Gathered values: 0 10 20 30.

Only the root process needs a valid receive array. recvcount is the
number of elements received from each process.

## 11. MPI_Reduce() and MPI_Allreduce() - Reduction

Purpose: Combine values contributed by all processes using an operation
such as sum, maximum, minimum or product.

Syntax: MPI_Reduce(sendbuf, recvbuf, count, datatype, operation, root,
communicator);

File: reduce.c

/\* Compile: mpicc reduce.c -o reduce

Run: mpirun -np 4 ./reduce \*/

``` c
#include <stdio.h>
```

``` c
#include <mpi.h>
```

int main(int argc, char \*argv\[\]) {

int rank, size;

int local_value, total = 0;

MPI_Init(&argc, &argv);

MPI_Comm_rank(MPI_COMM_WORLD, &rank);

MPI_Comm_size(MPI_COMM_WORLD, &size);

local_value = rank + 1;

MPI_Reduce(&local_value, &total, 1, MPI_INT,

MPI_SUM, 0, MPI_COMM_WORLD);

if (rank == 0)

printf("Sum of process values = %d`\n`{=tex}", total);

MPI_Finalize();

return 0;

}

Compile / run: mpicc reduce.c -o reduce \| mpirun -np 4 ./reduce

With 4 processes, local values are 1, 2, 3 and 4. Rank 0 prints: Sum of
process values = 10.

| Operation \| Meaning \| Example \|

| --- \| --- \| --- \|

| MPI_SUM \| Add all values \| 1 + 2 + 3 + 4 = 10 \|

| MPI_MAX \| Find maximum \| max(1,2,3,4) = 4 \|

| MPI_MIN \| Find minimum \| min(1,2,3,4) = 1 \|

| MPI_PROD \| Multiply values \| 1 × 2 × 3 × 4 = 24 \|

MPI_Allreduce performs a reduction and makes the final result available
to every process, whereas MPI_Reduce places the result only at the root.

## 12. MPI_Barrier() - Synchronization

MPI_Barrier makes each process wait until all processes in the
communicator have reached the barrier. Once all have arrived, they can
continue.

Syntax: MPI_Barrier(MPI_COMM_WORLD);

File: barrier.c

/\* Compile: mpicc barrier.c -o barrier

Run: mpirun -np 4 ./barrier \*/

``` c
#include <stdio.h>
```

``` c
#include <mpi.h>
```

int main(int argc, char \*argv\[\]) {

int rank;

MPI_Init(&argc, &argv);

MPI_Comm_rank(MPI_COMM_WORLD, &rank);

printf("Process %d reached the barrier`\n`{=tex}", rank);

MPI_Barrier(MPI_COMM_WORLD);

printf("Process %d passed the barrier`\n`{=tex}", rank);

MPI_Finalize();

return 0;

}

Compile / run: mpicc barrier.c -o barrier \| mpirun -np 4 ./barrier

The barrier guarantees that every process reached the synchronization
point before any process proceeds beyond it. It does not guarantee the
order of printed messages.

## 13. Sequential matrix addition (reference program)

Matrix addition means adding corresponding elements of two matrices of
the same dimensions. This sequential version is a reference for
understanding the work that will be divided among MPI processes.

Formula: C\[i\]\[j\] = A\[i\]\[j\] + B\[i\]\[j\]

/\* Compile: gcc matrix_add_serial.c -o matrix_add_serial

Run: ./matrix_add_serial \*/

``` c
#include <stdio.h>
```

``` c
#include <stdlib.h>
```

int main(void) {

int n;

printf("Enter matrix order:");

scanf("%d", &n);

if (n \<= 0) return 1;

int *A = malloc(n * n \* sizeof(int));

int *B = malloc(n * n \* sizeof(int));

int *C = malloc(n * n \* sizeof(int));

if (!A \|\| !B \|\| !C) return 1;

printf("Enter matrix A elements:`\n`{=tex}");

for (int i = 0; i \< n\*n; i++) scanf("%d", &A\[i\]);

printf("Enter matrix B elements:`\n`{=tex}");

for (int i = 0; i \< n\*n; i++) scanf("%d", &B\[i\]);

for (int i = 0; i \< n\*n; i++) C\[i\] = A\[i\] + B\[i\];

printf("Result matrix C:`\n`{=tex}");

for (int i = 0; i \< n; i++) {

for (int j = 0; j \< n; j++) printf("%d", C\[i\*n+j\]);

printf("`\n`{=tex}");

}

free(A); free(B); free(C);

return 0;

}

## 14. Parallel matrix addition using MPI

Idea: Rank 0 reads the two matrices. MPI_Bcast shares the matrix order
with all processes. MPI_Scatter distributes equal numbers of elements to
each process. Every process adds its local elements. MPI_Gather collects
the partial result at rank 0.

This practical version requires n\*n to be divisible into equal chunks,
so it checks that n is divisible by the number of processes (each
process receives the same number of complete rows).

/\* Compile: mpicc matrix_add_mpi.c -o matrix_add_mpi

Run: mpirun -np 4 ./matrix_add_mpi \*/

``` c
#include <stdio.h>
```

``` c
#include <stdlib.h>
```

``` c
#include <mpi.h>
```

int main(int argc, char \*argv\[\]) {

int rank, size, n;

int *A = NULL, *B = NULL, \*C = NULL;

int *localA, *localB, \*localC;

MPI_Init(&argc, &argv);

MPI_Comm_rank(MPI_COMM_WORLD, &rank);

MPI_Comm_size(MPI_COMM_WORLD, &size);

n = 0;

if (rank == 0) {

printf("Enter matrix order n (must be divisible by %d):", size);

fflush(stdout);

scanf("%d", &n);

}

MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

if (n \<= 0 \|\| n % size != 0) {

if (rank == 0) printf("Invalid n. Use a positive n divisible by process
count.`\n`{=tex}");

MPI_Finalize();

return 0;

}

int total = n \* n;

int local_count = total / size;

localA = malloc(local_count \* sizeof(int));

localB = malloc(local_count \* sizeof(int));

localC = malloc(local_count \* sizeof(int));

if (!localA \|\| !localB \|\| !localC) MPI_Abort(MPI_COMM_WORLD, 1);

if (rank == 0) {

A = malloc(total \* sizeof(int));

B = malloc(total \* sizeof(int));

C = malloc(total \* sizeof(int));

if (!A \|\| !B \|\| !C) MPI_Abort(MPI_COMM_WORLD, 1);

printf("Enter matrix A elements (row-wise):`\n`{=tex}");

for (int i = 0; i \< total; i++) scanf("%d", &A\[i\]);

printf("Enter matrix B elements (row-wise):`\n`{=tex}");

for (int i = 0; i \< total; i++) scanf("%d", &B\[i\]);

}

MPI_Scatter(A, local_count, MPI_INT, localA, local_count,

MPI_INT, 0, MPI_COMM_WORLD);

MPI_Scatter(B, local_count, MPI_INT, localB, local_count,

MPI_INT, 0, MPI_COMM_WORLD);

for (int i = 0; i \< local_count; i++)

localC\[i\] = localA\[i\] + localB\[i\];

MPI_Gather(localC, local_count, MPI_INT, C, local_count,

MPI_INT, 0, MPI_COMM_WORLD);

if (rank == 0) {

printf("Result matrix C:`\n`{=tex}");

for (int i = 0; i \< n; i++) {

for (int j = 0; j \< n; j++) printf("%d", C\[i\*n+j\]);

printf("`\n`{=tex}");

}

free(A); free(B); free(C);

}

free(localA); free(localB); free(localC);

MPI_Finalize();

return 0;

}

### Example

For a small test, run with 2 processes and n = 2. Enter A = \[1 2; 3 4\]
and B = \[5 6; 7 8\]. The result is \[6 8; 10 12\]. For a 4-process run,
choose n = 4 so each process receives one row.

For n=4 and 4 processes, each process receives one row from A and one
row from B, computes that row of C, and the root gathers all rows.

MPI functions used: MPI_Bcast shares n; MPI_Scatter distributes matrix
elements; each process performs local addition; MPI_Gather assembles C
on rank 0.

## 15. Dot product using MPI

The dot product of two vectors A and B of length n is the sum of the
products of corresponding elements.

Formula: dot = A\[0\]*B\[0\] + A\[1\]*B\[1\] + ... + A\[n-1\]\*B\[n-1\].

MPI approach: rank 0 reads both vectors, MPI_Bcast shares n, MPI_Scatter
distributes equal chunks, each process calculates a partial dot product,
and MPI_Reduce adds all partial results.

This simple practical requires n to be divisible by the number of
processes.

/\* Compile: mpicc dot_product_mpi.c -o dot_product_mpi

Run: mpirun -np 4 ./dot_product_mpi \*/

``` c
#include <stdio.h>
```

``` c
#include <stdlib.h>
```

``` c
#include <mpi.h>
```

int main(int argc, char \*argv\[\]) {

int rank, size, n;

int *A = NULL, *B = NULL;

int *localA, *localB;

long long local_dot = 0, global_dot = 0;

MPI_Init(&argc, &argv);

MPI_Comm_rank(MPI_COMM_WORLD, &rank);

MPI_Comm_size(MPI_COMM_WORLD, &size);

n = 0;

if (rank == 0) {

printf("Enter vector size n (must be divisible by %d):", size);

fflush(stdout);

scanf("%d", &n);

}

MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

if (n \<= 0 \|\| n % size != 0) {

if (rank == 0) printf("Invalid n. Use a positive n divisible by process
count.`\n`{=tex}");

MPI_Finalize();

return 0;

}

int local_n = n / size;

localA = malloc(local_n \* sizeof(int));

localB = malloc(local_n \* sizeof(int));

if (!localA \|\| !localB) MPI_Abort(MPI_COMM_WORLD, 1);

if (rank == 0) {

A = malloc(n \* sizeof(int));

B = malloc(n \* sizeof(int));

if (!A \|\| !B) MPI_Abort(MPI_COMM_WORLD, 1);

printf("Enter A elements:");

for (int i = 0; i \< n; i++) scanf("%d", &A\[i\]);

printf("Enter B elements:");

for (int i = 0; i \< n; i++) scanf("%d", &B\[i\]);

}

MPI_Scatter(A, local_n, MPI_INT, localA, local_n,

MPI_INT, 0, MPI_COMM_WORLD);

MPI_Scatter(B, local_n, MPI_INT, localB, local_n,

MPI_INT, 0, MPI_COMM_WORLD);

for (int i = 0; i \< local_n; i++)

local_dot += (long long)localA\[i\] \* localB\[i\];

MPI_Reduce(&local_dot, &global_dot, 1, MPI_LONG_LONG_INT,

MPI_SUM, 0, MPI_COMM_WORLD);

if (rank == 0)

printf("Dot product = %lld`\n`{=tex}", global_dot);

free(localA); free(localB);

if (rank == 0) { free(A); free(B); }

MPI_Finalize();

return 0;

}

### Example

For A = \[1, 2, 3, 4\] and B = \[5, 6, 7, 8\], the dot product is 1*5 +
2*6 + 3*7 + 4*8 = 70. With 2 processes, each process handles 2 pairs,
calculates a partial sum, and MPI_Reduce adds the partial sums.

## 16. Important differences and common mistakes

| Compare \| Difference \|

| --- \| --- \|

| Point-to-point vs collective \| Point-to-point communicates between a
  sender and receiver; collective operations involve all processes in a
  communicator. \|

| Send vs Bcast \| Send targets one destination; Bcast sends the same
  data from a root to all processes. \|

| Scatter vs Bcast \| Scatter gives different chunks; Bcast gives the
  same data. \|

| Gather vs Reduce \| Gather collects each process contribution; Reduce
  combines the contributions using an operation. \|

| Reduce vs Allreduce \| Reduce gives the result to the root; Allreduce
  gives the result to all processes. \|

| Group vs communicator \| A group is an ordered process set; a
  communicator adds the communication context. \|

### Common mistakes to avoid

-   Forgetting MPI_Init or MPI_Finalize.

-   Using the wrong rank as source, destination or root.

-   Using different tags or communicators in a send/receive pair.

-   Calling a collective on only one process instead of all processes in
    the communicator.

-   Using incompatible datatypes or counts between sender and receiver.

-   Assuming output from multiple processes will appear in rank order.

-   Forgetting that the matrix addition and dot product examples require
    input size divisible by process count.

-   Using mpicc incorrectly: compile with mpicc, not plain gcc, unless
    MPI include and library flags are manually supplied.

## 17. Viva questions and quick revision

What is MPI? MPI is a standard interface for communication between
processes in parallel programs.

What is Open MPI? It is an implementation of the MPI standard.

What is mpicc? A compiler wrapper that compiles C programs with MPI
headers and libraries.

What is mpirun? A launcher used to start an MPI program with a specified
number of processes.

What is a process? An independent running instance of the MPI program.

What is rank? The unique process ID inside a communicator.

What is MPI_COMM_WORLD? The default communicator containing all launched
MPI processes.

What is a group? An ordered set of processes.

What is a communicator? A process group together with a communication
context.

What is point-to-point communication? Direct communication between a
sender and receiver.

What is MPI_Send? A blocking send routine.

What is MPI_Recv? A blocking receive routine.

What is a tag? An integer label used to distinguish or match messages.

What is broadcast? Sending the same data from one root process to all
processes.

What is scatter? Distributing different portions of root data to
processes.

What is gather? Collecting contributions from all processes at the root.

What is reduce? Combining values contributed by processes using an
operation.

What is MPI_SUM? A predefined reduction operation that adds values.

What is MPI_Barrier? A synchronization call where all processes wait
until all have arrived.

What is the difference between MPI_Reduce and MPI_Allreduce? Reduce
returns the result at the root; Allreduce returns it to all processes.

Which functions are used in MPI matrix addition? MPI_Bcast, MPI_Scatter
and MPI_Gather, plus local addition on each process.

Which function combines partial dot products? MPI_Reduce with MPI_SUM.

How do you compile an MPI C program? mpicc file.c -o file.

How do you run it with 4 processes? mpirun -np 4 ./file.

## One-minute final revision

MPI = Message Passing Interface. Process = running instance. Rank =
process ID. Group = ordered process set. Communicator = group +
communication context. Send/Recv = one-to-one. Bcast = same data to all.
Scatter = distribute chunks. Gather = collect chunks. Reduce = combine
values at root. Allreduce = combine and share with all. Barrier =
synchronization.

Standard MPI flow: MPI_Init -\> MPI_Comm_rank / MPI_Comm_size -\>
communication and computation -\> MPI_Finalize.
