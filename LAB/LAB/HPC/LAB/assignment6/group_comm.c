#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int world_rank;
    int world_size;

    MPI_Group world_group;
    MPI_Group selected_group;

    MPI_Comm selected_comm = MPI_COMM_NULL;

    int selected_ranks[] = {0, 2, 4, 6};

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 8)
    {
        if (world_rank == 0)
        {
            printf("Run this program with at least 8 processes.\n");
        }

        MPI_Finalize();
        return 0;
    }

    MPI_Comm_group(
        MPI_COMM_WORLD,
        &world_group
    );

    MPI_Group_incl(
        world_group,
        4,
        selected_ranks,
        &selected_group
    );

    MPI_Comm_create_group(
        MPI_COMM_WORLD,
        selected_group,
        0,
        &selected_comm
    );

    if (selected_comm != MPI_COMM_NULL)
    {
        int local_rank;
        int local_size;

        MPI_Comm_rank(
            selected_comm,
            &local_rank
        );

        MPI_Comm_size(
            selected_comm,
            &local_size
        );

        printf(
            "Global rank = %d, subgroup rank = %d, subgroup size = %d\n",
            world_rank,
            local_rank,
            local_size
        );

        MPI_Comm_free(&selected_comm);
    }

    MPI_Group_free(&selected_group);
    MPI_Group_free(&world_group);

    MPI_Finalize();

    return 0;
}
