#include <mpi.h>
#include <stdio.h>
#include <limits.h>

int main(int argc, char *argv[])
{
    int rank;
    int size;

    int data[5] = {10, 20, 30, 40, 50};
    int received[5];

    int sum = 0;
    int max = INT_MIN;

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
            data,
            5,
            MPI_INT,
            1,
            300,
            MPI_COMM_WORLD
        );

        printf(
            "Process 0 sent: 10 20 30 40 50\n"
        );
    }

    else if (rank == 1)
    {
        MPI_Recv(
            received,
            5,
            MPI_INT,
            0,
            300,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        for (int i = 0; i < 5; i++)
        {
            sum += received[i];

            if (received[i] > max)
            {
                max = received[i];
            }
        }

        printf("Process 1 received: ");

        for (int i = 0; i < 5; i++)
        {
            printf("%d ", received[i]);
        }

        printf("\n");

        printf("Sum = %d\n", sum);

        printf(
            "Average = %.2f\n",
            sum / 5.0
        );

        printf(
            "Maximum = %d\n",
            max
        );
    }

    MPI_Finalize();

    return 0;
}
