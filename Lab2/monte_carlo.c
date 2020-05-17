#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    clock_t start = clock();
    int car_num = atoi(argv[1]);
    int period = atoi(argv[2]);
    int v_max = atoi(argv[3]);
    float p = atof(argv[4]);
    int car_v[1000000];
    int car_pos[1000000];

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int last_rank = (world_rank - 1 + world_size) % world_size;
    int next_rank = (world_rank + 1) % world_size;
    
    // datatype
    int valid_car_num;
    valid_car_num = (car_num + world_size - 1) / world_size;
    
    MPI_Datatype intvector;
    MPI_Type_vector(valid_car_num, 1, world_size, MPI_INT, &intvector);
    MPI_Type_commit(&intvector);

    // init
    for(int car_i = world_rank; car_i < car_num; car_i += world_size) {
        car_v[car_i] = 0;
        car_pos[car_i] = car_i;
    }

    // period 0
    MPI_Send(car_pos + world_rank, 1, intvector, last_rank, 0, MPI_COMM_WORLD);

    // period 0+
    for(int period_i = 1; period_i <= period; period_i++) {
        printf("%d %d\n", world_rank, period_i);
        MPI_Recv(car_pos + next_rank, 1, intvector, next_rank, period_i - 1, 
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int car_i = world_rank; car_i < car_num; car_i += world_size) {
            int d;
            if(car_i < car_num - 1) 
                d = car_pos[car_i+1] - car_pos[car_i];
            // (2)
            if(car_i == car_num - 1 || (d - 1) > car_v[car_i]) 
                car_v[car_i]++;
            // (3)
            else if(d <= car_v[car_i])
                car_v[car_i] = d - 1;
            // (4)
            if(car_v[car_i] > 0 && (double)rand()/(double)RAND_MAX < p)
                car_v[car_i] -= 1;
            // (5)
            car_pos[car_i] += car_v[car_i];
        }
        MPI_Send(car_pos + world_rank, 1, intvector, last_rank, period_i, MPI_COMM_WORLD);
    }

    int tag_v = period + 1;
    int tag_pos = period + 2;
    if(world_rank == 0) {
        printf("Road state: \n");
        for(int rank = 1; rank < world_size; rank++) {
            MPI_Recv(car_v + rank, 1, intvector, rank, tag_v, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(car_pos + rank, 1, intvector, rank, tag_pos, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for(int car_i = 0; car_i < car_num; car_i++) 
            printf("Car%d: v=%d pos=%d\n", car_i, car_v[car_i], car_pos[car_i]);
        printf("Time: %f\n", (clock() - start) / (float)CLOCKS_PER_SEC);
    } else {
        MPI_Send(car_v + world_rank, 1, intvector, 0, tag_v, MPI_COMM_WORLD);
        MPI_Send(car_pos + world_rank, 1, intvector, 0, tag_pos, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}