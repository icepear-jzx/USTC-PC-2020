#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double soft2 = 1e-18;
double m = 1e4;
double G = 6.67e11;
double local_x[1024];
double local_y[1024];
double local_vx[1024];
double local_vy[1024];
double local_fx[1024];
double local_fy[1024];
double global_x[1024];
double global_y[1024];

double dist2(int i, int j) {
    double x1 = global_x[i];
    double y1 = global_y[i];
    double x2 = global_x[j];
    double y2 = global_y[j];
    return pow(x1 - x2, 2) + pow(y1 - y2, 2);
}

void compute_force(int body_rank_start, int body_rank_end, int body_num) {
    for (int i = body_rank_start; i < body_rank_end; i++) {
        local_fx[i] = 0;
        local_fy[i] = 0;
        for (int j = 0; j < body_num; j++) {
            double denominator = pow(dist2(i, j) + soft2, 1.5);
            local_fx[i] += (global_x[j] - global_x[i]) / denominator;
            local_fy[i] += (global_y[j] - global_y[i]) / denominator;
        }
        local_fx[i] *= G * m * m;
        local_fy[i] *= G * m * m;
    }
}

void compute_velocities(int body_rank_start, int body_rank_end) {
    for (int i = body_rank_start; i < body_rank_end; i++) {
        local_vx[i] += local_fx[i] / m;
        local_vy[i] += local_fy[i] / m;
    }
}

void compute_positions(int body_rank_start, int body_rank_end) {
    for (int i = body_rank_start; i < body_rank_end; i++) {
        local_x[i] += local_vx[i];
        local_y[i] += local_vy[i];
    }
}

void print_status(int world_rank, int body_rank_start, int body_rank_end) {
    for (int i = body_rank_start; i < body_rank_end; i++) {
        printf("Node%d Body%d: fx=%e, fy=%e, vx=%e, vy=%e, x=%e, y=%e\n", 
            world_rank, i, local_fx[i], local_fy[i], local_vx[i], 
            local_vy[i], local_x[i], local_y[i]);
    }
}

int main(int argc, char *argv[]) {
    double t1, t2;
    int body_num = atoi(argv[1]);
    int periods = atoi(argv[2]);
    int sqrt_body_num = ceil(sqrt(body_num));

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int base_body_num = (body_num + world_size - 1) / world_size;
    int last_body_num = body_num - base_body_num * (world_size - 1);
    int valid_body_num;
    if(world_rank + 1 == world_size)
        valid_body_num = last_body_num;
    else
        valid_body_num = base_body_num;
    
    int body_rank_start = world_rank * base_body_num;
    int body_rank_end = body_rank_start + valid_body_num;

    // init
    for (int i = body_rank_start; i < body_rank_end; i++) {
        local_x[i] = (i / sqrt_body_num) * 1e-2;
        local_y[i] = (i % sqrt_body_num) * 1e-2;
        local_vx[i] = 0;
        local_vy[i] = 0;
        local_fx[i] = 0;
        local_fy[i] = 0;
    }

    if(world_rank == 0) {
        t1 = MPI_Wtime();
    }

    for (int period = 0; period < periods; period++) {

        MPI_Allgather(&local_x[body_rank_start], base_body_num, MPI_DOUBLE,
            global_x, base_body_num, MPI_DOUBLE, MPI_COMM_WORLD);
        // printf("Node%d-Period%d: entering barrier.\n", world_rank, period);
        MPI_Barrier(MPI_COMM_WORLD);
        // printf("Node%d-Period%d: leaving barrier.\n", world_rank, period);

        MPI_Allgather(&local_y[body_rank_start], base_body_num, MPI_DOUBLE,
            global_y, base_body_num, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        // if (world_rank == 0) {
        //     for (int i = 0; i < body_num; i++) {
        //         printf("i=%d, x=%f, y=%f\n", i, global_x[i], global_y[i]);
        //     }
        // }

        // print_status(world_rank, body_rank_start, body_rank_end);

        compute_force(body_rank_start, body_rank_end, body_num);
        compute_velocities(body_rank_start, body_rank_end);
        compute_positions(body_rank_start, body_rank_end);
    }

    if (world_rank == 0) {
        t2 = MPI_Wtime();
        printf("Time: %lf\n", t2 - t1);
    }

    // print_status(world_rank, body_rank_start, body_rank_end);
    
    MPI_Finalize();

    return 0;
}