#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int is_prime(int num) {
    for(int i = 2; i <= sqrt(num); i++) {
        if(num % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    double t1, t2;
    int n = atoi(argv[1]);

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(world_rank == 0) {
        t1 = MPI_Wtime();
    }

    int local_cnt = 0;
    for(int i = 2 + world_rank; i <= n; i += world_size) {
        if(is_prime(i)) local_cnt++;
    }

    int global_cnt;
    MPI_Reduce(&local_cnt, &global_cnt, 1, MPI_INT, MPI_SUM, 0,
            MPI_COMM_WORLD);

    if(world_rank == 0) {
        t2 = MPI_Wtime();
        printf("Prime number: %d\n", global_cnt);
        printf("Time: %lf\n", t2 - t1);
    }

    MPI_Finalize();

    return 0;
}