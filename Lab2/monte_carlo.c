#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int car_v[1000000];
int car_pos[1000000];
int car_v_recv[1000000];
int car_pos_recv[1000000];

int main(int argc, char *argv[]) {
    double t1, t2;
    int car_num = atoi(argv[1]);
    int period = atoi(argv[2]);
    int v_max = atoi(argv[3]);
    float p = atof(argv[4]);
    int next_car_pos;

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int last_rank = world_rank - 1;
    int next_rank = world_rank + 1;

    int base_car_num = (car_num + world_size - 1) / world_size;
    int last_car_num = car_num - base_car_num * (world_size - 1);
    int valid_car_num;
    if(next_rank == world_size)
        valid_car_num = last_car_num;
    else
        valid_car_num = base_car_num;
    
    // init
    for(int car_i = 0; car_i < valid_car_num; car_i++) {
        car_v[car_i] = 0;
        car_pos[car_i] = base_car_num * world_rank + car_i;
    }

    if(world_rank == 0) {
        t1 = MPI_Wtime();
    }

    // period 0
    if(last_rank >= 0) {
        MPI_Send(car_pos, 1, MPI_INT, last_rank, 0, MPI_COMM_WORLD);
    }

    // period 0+
    for(int period_i = 1; period_i <= period; period_i++) {
        // printf("%d %d\n", world_rank, period_i);
        if(next_rank < world_size) {
            MPI_Recv(&next_car_pos, 1, MPI_INT, next_rank, period_i - 1, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for(int car_i = 0; car_i < valid_car_num; car_i++) {
            int d;
            if(car_i < valid_car_num - 1) 
                d = car_pos[car_i+1] - car_pos[car_i];
            else if(next_rank == world_size)
                d = v_max + 1;
            else
                d = next_car_pos - car_pos[car_i];
            // (2)
            if(car_num == base_car_num * world_rank + car_i || (d - 1) > car_v[car_i]) {
                if(car_v[car_i] < v_max)
                    car_v[car_i]++;
            }
            // (3)
            else {
                car_v[car_i] = d - 1;
            }
            // (4)
            if(car_v[car_i] > 0 && (double)rand()/(double)RAND_MAX < p)
                car_v[car_i] -= 1;
            // (5)
            car_pos[car_i] += car_v[car_i];
        }
        if(last_rank >= 0) {
            MPI_Send(car_pos, 1, MPI_INT, last_rank, period_i, MPI_COMM_WORLD);
        }
    }

    int tag_v = period + 1;
    int tag_pos = period + 2;
    if(world_rank == 0) {
        // printf("Road state: \n");
        // for(int car_i = 0; car_i < valid_car_num; car_i++) {
        //     printf("Car%d: v=%d pos=%d\n", car_i, 
        //         car_v[car_i], car_pos[car_i]);
        // }
        for(int rank = 1; rank < world_size; rank++) {
            if(rank + 1 == world_size) {
                MPI_Recv(car_v_recv, last_car_num, MPI_INT, rank, tag_v, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(car_pos_recv, last_car_num, MPI_INT, rank, tag_pos, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // for(int car_i = 0; car_i < last_car_num; car_i++) {
                //     printf("Car%d: v=%d pos=%d\n", base_car_num * rank + car_i, 
                //         car_v_recv[car_i], car_pos_recv[car_i]);
                // }
            }
            else {
                MPI_Recv(car_v_recv, base_car_num, MPI_INT, rank, tag_v, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(car_pos_recv, base_car_num, MPI_INT, rank, tag_pos, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // for(int car_i = 0; car_i < base_car_num; car_i++) {
                //     printf("Car%d: v=%d pos=%d\n", base_car_num * rank + car_i, 
                //         car_v_recv[car_i], car_pos_recv[car_i]);
                // }
            }
        }
        t2 = MPI_Wtime();
        printf("Time: %lf\n", t2 - t1);
    } else {
        MPI_Send(car_v, valid_car_num, MPI_INT, 0, tag_v, MPI_COMM_WORLD);
        MPI_Send(car_pos, valid_car_num, MPI_INT, 0, tag_pos, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}