#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

double soft2 = 1e-18;
double m = 1e4;
double G = 6.67e11;
double bodies_x[1024];
double bodies_y[1024];
double bodies_vx[1024];
double bodies_vy[1024];
double bodies_fx[1024];
double bodies_fy[1024];

double dist2(int i, int j) {
    double x1 = bodies_x[i];
    double y1 = bodies_y[i];
    double x2 = bodies_x[j];
    double y2 = bodies_y[j];
    return pow(x1 - x2, 2) + pow(y1 - y2, 2);
}

void compute_force(int body_rank_start, int body_rank_end, int body_num) {
    for (int i = body_rank_start; i < body_rank_end; i++) {
        bodies_fx[i] = 0;
        bodies_fy[i] = 0;
        for (int j = 0; j < body_num; j++) {
            double denominator = pow(dist2(i, j) + soft2, 1.5);
            bodies_fx[i] += (bodies_x[j] - bodies_x[i]) / denominator;
            bodies_fy[i] += (bodies_y[j] - bodies_y[i]) / denominator;
        }
        bodies_fx[i] *= G * m * m;
        bodies_fy[i] *= G * m * m;
    }
}

void compute_velocities(int body_rank_start, int body_rank_end) {
    for (int i = body_rank_start; i < body_rank_end; i++) {
        bodies_vx[i] += bodies_fx[i] / m;
        bodies_vy[i] += bodies_fy[i] / m;
    }
}

void compute_positions(int body_rank_start, int body_rank_end) {
    for (int i = body_rank_start; i < body_rank_end; i++) {
        bodies_x[i] += bodies_vx[i];
        bodies_y[i] += bodies_vy[i];
    }
}

void print_status(int world_rank, int body_rank_start, int body_rank_end) {
    for (int i = body_rank_start; i < body_rank_end; i++) {
        printf("Node%d Body%d: fx=%e, fy=%e, vx=%e, vy=%e, x=%e, y=%e\n", 
            world_rank, i, bodies_fx[i], bodies_fy[i], bodies_vx[i], 
            bodies_vy[i], bodies_x[i], bodies_y[i]);
    }
}

int main(int argc, char *argv[]) {
    int body_num = atoi(argv[1]);
    int periods = atoi(argv[2]);
    int sqrt_body_num = ceil(sqrt(body_num));

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int base_body_num = ceil((double)body_num / (double)world_size);
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
        bodies_x[i] = (i / sqrt_body_num) * 1e-2;
        bodies_y[i] = (i % sqrt_body_num) * 1e-2;
        bodies_vx[i] = 0;
        bodies_vy[i] = 0;
        bodies_fx[i] = 0;
        bodies_fy[i] = 0;
    }

    for (int period = 0; period < periods; period++) {

        MPI_Allgather(MPI_IN_PLACE, base_body_num, MPI_DOUBLE,
            bodies_x, body_num, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Allgather(MPI_IN_PLACE, base_body_num, MPI_DOUBLE,
            bodies_y, body_num, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        // print_status(world_rank, body_rank_start, body_rank_end);

        compute_force(body_rank_start, body_rank_end, body_num);
        compute_velocities(body_rank_start, body_rank_end);
        compute_positions(body_rank_start, body_rank_end);
    }

    // print_status(world_rank, body_rank_start, body_rank_end);

    return 0;
}