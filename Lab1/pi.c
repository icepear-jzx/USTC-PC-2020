#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    clock_t start = clock();
    int n = atoi(argv[1]);

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    double local_sum = 0;
    for(int i = world_rank; i < n; i += world_size) {
        if(i % 2 == 0)
            local_sum += (double)1 / (double)(2*i+1);
        else
            local_sum -= (double)1 / (double)(2*i+1);
    }

    double global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
            MPI_COMM_WORLD);

    if(world_rank == 0) {
        printf("PI = %f\n", global_sum * 4);
        printf("Time: %f\n", (clock() - start) / (float)CLOCKS_PER_SEC);
    }

    MPI_Finalize();

    return 0;
}