#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int rank;
    int size;

    int sizes[] = {
        1024,
        10 * 1024,
        100 * 1024,
        1024 * 1024,
        10 * 1024 * 1024
    };

    int iterations = 1000;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2)
    {
        if (rank == 0)
        {
            printf("Run with at least 2 processes.\n");
        }

        MPI_Finalize();
        return 0;
    }

    for (int s = 0; s < 5; s++)
    {
        int bytes = sizes[s];

        char *buffer =
            malloc(bytes);

        MPI_Barrier(MPI_COMM_WORLD);

        double start = MPI_Wtime();

        if (rank == 0)
        {
            for (int i = 0; i < iterations; i++)
            {
                MPI_Send(
                    buffer,
                    bytes,
                    MPI_CHAR,
                    1,
                    500,
                    MPI_COMM_WORLD
                );

                MPI_Recv(
                    buffer,
                    bytes,
                    MPI_CHAR,
                    1,
                    501,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );
            }
        }

        else if (rank == 1)
        {
            for (int i = 0; i < iterations; i++)
            {
                MPI_Recv(
                    buffer,
                    bytes,
                    MPI_CHAR,
                    0,
                    500,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );

                MPI_Send(
                    buffer,
                    bytes,
                    MPI_CHAR,
                    0,
                    501,
                    MPI_COMM_WORLD
                );
            }
        }

        double end = MPI_Wtime();

        if (rank == 0)
        {
            double total_time =
                end - start;

            double rtt =
                total_time / iterations;

            printf(
                "%d bytes: iterations=%d "
                "total=%.9f s "
                "average RTT=%.9f s\n",
                bytes,
                iterations,
                total_time,
                rtt
            );
        }

        free(buffer);
    }

    MPI_Finalize();

    return 0;
}
