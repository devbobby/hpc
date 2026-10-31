#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int rank;
    int size;

    int value = 100;
    int received = 0;

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

    if (rank == 0)
    {
        MPI_Send(
            &value,
            1,
            MPI_INT,
            1,
            200,
            MPI_COMM_WORLD
        );

        printf("Process 0 sent value = %d\n", value);
    }

    else if (rank == 1)
    {
        MPI_Recv(
            &received,
            1,
            MPI_INT,
            0,
            200,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        printf(
            "Process 1 received value = %d\n",
            received
        );
    }

    MPI_Finalize();

    return 0;
}
