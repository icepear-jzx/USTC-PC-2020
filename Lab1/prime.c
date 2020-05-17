#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int is_prime(int num) {
    for(int i = 2; i <= sqrt(num); i++) {
        if(num % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    clock_t start = clock();
    int n = atoi(argv[1]);

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int local_cnt = 0;
    for(int i = 2 + world_rank; i <= n; i += world_size) {
        if(is_prime(i)) local_cnt++;
    }

    int global_cnt;
    MPI_Reduce(&local_cnt, &global_cnt, 1, MPI_INT, MPI_SUM, 0,
            MPI_COMM_WORLD);

    if(world_rank == 0) {
        printf("Prime number: %d\n", global_cnt);
        printf("Time: %f\n", (clock() - start) / (float)CLOCKS_PER_SEC);
    }

    MPI_Finalize();

    return 0;
}