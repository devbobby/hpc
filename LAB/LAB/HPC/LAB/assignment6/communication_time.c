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

    int num_sizes = 5;

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

    for (int s = 0; s < num_sizes; s++)
    {
        int bytes = sizes[s];

        char *buffer =
            (char *)malloc(bytes);

        if (rank == 0)
        {
            for (int i = 0; i < bytes; i++)
            {
                buffer[i] = 'A';
            }

            MPI_Barrier(MPI_COMM_WORLD);

            double start = MPI_Wtime();

            MPI_Send(
                buffer,
                bytes,
                MPI_CHAR,
                1,
                400 + s,
                MPI_COMM_WORLD
            );

            double end = MPI_Wtime();

            printf(
                "%d bytes, send time = %.9f seconds\n",
                bytes,
                end - start
            );
        }

        else if (rank == 1)
        {
            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Recv(
                buffer,
                bytes,
                MPI_CHAR,
                0,
                400 + s,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );
        }

        free(buffer);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}


