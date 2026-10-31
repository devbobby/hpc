#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
    int rank;
    int size;

    char message[] = "Hello from Process 0";
    char received[100];

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
            message,
            strlen(message) + 1,
            MPI_CHAR,
            1,
            100,
            MPI_COMM_WORLD
        );

        printf("Process 0 sent: %s\n", message);
    }

    else if (rank == 1)
    {
        MPI_Recv(
            received,
            100,
            MPI_CHAR,
            0,
            100,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        printf("Process 1 received: %s\n", received);
    }

    MPI_Finalize();

    return 0;
}
