#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define INF 2147483647

int A[10000000];


void merge(int *A, int l, int m, int r){
    int i, j, k, n1 = m - l + 1, n2 = r - m;
    int *L = (int *)malloc((n1 + 1) * sizeof(int));
    int *R = (int *)malloc((n2 + 1) * sizeof(int));
    for(i = 0; i < n1; i++) L[i] = A[l + i];
    for(j = 0; j < n2; j++) R[j] = A[m + 1 + j];
    L[i] = R[j] = INF;
    i = j = 0;
    for(k = l; k <= r; k++) {
        if(L[i] <= R[j]) 
            A[k] = L[i++]; 
        else 
            A[k] = R[j++];
    }
    free(L); free(R);
} 


void merge_sort(int *A, int l, int r){
    if(l < r){
        int m = (l + r) / 2;
        merge_sort(A, l, m);
        merge_sort(A, m + 1, r);
        merge(A, l, m, r);
    }
} 


void psrs(int n, int world_rank, int world_size){
    int per; 
    int *samples, *global_samples;
    int *pivots; 
    int *sizes, *newsizes;
    int *offsets, *newoffsets;
    int *newdatas;
    int newdatassize;
    int *global_sizes;
    int *global_offsets;

    per = n / world_size;
    samples = (int *)malloc(world_size * sizeof(int));
    pivots = (int *)malloc(world_size * sizeof(int));
    if(world_rank == 0){
        global_samples = (int *)malloc(world_size * world_size * sizeof(int)); 
    }

    // MPI_Barrier(MPI_COMM_WORLD);

    merge_sort(A, world_rank * per, (world_rank + 1) * per - 1); 

    for(int k = 0; k < world_size; k++) 
        samples[k] = A[world_rank * per + k * per / world_size];

    MPI_Gather(samples, world_size, MPI_INT, global_samples, world_size, MPI_INT, 0, MPI_COMM_WORLD);

    if(world_rank == 0){
        merge_sort(global_samples, 0, world_size * world_size - 1);
        for(int k = 0; k < world_size - 1; k++)
            pivots[k] = global_samples[(k + 1) * world_size];
        pivots[world_size - 1] = INF;
    }

    MPI_Bcast(pivots, world_size, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Barrier(MPI_COMM_WORLD);

    sizes = (int *)calloc(world_size, sizeof(int));
    offsets = (int *)calloc(world_size, sizeof(int));
    newsizes = (int *)calloc(world_size, sizeof(int));
    newoffsets = (int *)calloc(world_size, sizeof(int));

    for(int k = 0, j = world_rank * per; j < world_rank * per + per; j++){ 
        if(A[j] < pivots[k])
            sizes[k]++;
        else
            sizes[++k]++;
    }

    MPI_Alltoall(sizes, 1, MPI_INT, newsizes, 1, MPI_INT, MPI_COMM_WORLD);

    newdatassize = newsizes[0];
    for(int k = 1; k < world_size; k++){
        offsets[k] = offsets[k - 1] + sizes[k - 1];
        newoffsets[k] = newoffsets[k - 1] + newsizes[k - 1];
        newdatassize += newsizes[k];
    }

    newdatas = (int *)malloc(newdatassize * sizeof(int)); 

    MPI_Alltoallv(&(A[world_rank * per]), sizes, offsets, MPI_INT, newdatas, newsizes, newoffsets, MPI_INT, MPI_COMM_WORLD);
    // MPI_Barrier(MPI_COMM_WORLD);

    merge_sort(newdatas, 0, newdatassize - 1);
    // MPI_Barrier(MPI_COMM_WORLD);

    if(world_rank == 0)
        global_sizes = (int *)calloc(world_size, sizeof(int));
    MPI_Gather(&newdatassize, 1, MPI_INT, global_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(world_rank == 0){ 
        global_offsets = (int *)calloc(world_size, sizeof(int));
        for(int k = 1; k < world_size; k++)
            global_offsets[k] = global_offsets[k - 1] + global_sizes[k - 1];
    }

    MPI_Gatherv(newdatas, newdatassize, MPI_INT, A, global_sizes, global_offsets, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Barrier(MPI_COMM_WORLD);

    free(samples); samples = NULL;
    free(pivots); pivots = NULL;
    free(sizes); sizes = NULL;
    free(offsets); offsets = NULL;
    free(newdatas); newdatas = NULL;
    free(newsizes); newsizes = NULL;
    free(newoffsets); newoffsets = NULL;
    if(world_rank == 0){ 
        free(global_samples); global_samples = NULL;
        free(global_sizes); global_sizes = NULL;
        free(global_offsets); global_offsets = NULL;
    }
}


int main(int argc, char *argv[]){
    int array_len = atoi(argv[1]);
    int num_range = atoi(argv[2]);
    double t1, t2;

    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    if(world_rank == 0) {
        for (int i = 0; i < array_len; i++) {
            A[i] = rand() % num_range;
        }
    }
    MPI_Bcast(A, 10000000, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Barrier(MPI_COMM_WORLD);

    if(world_rank == 0) {
        t1 = MPI_Wtime();
    }

    psrs(array_len, world_rank, world_size);

    if(world_rank == 0) {
        t2 = MPI_Wtime();
        for(int i = 0; i < array_len; i++)
            printf("%d ", A[i]);
        printf("\n");
        printf("time: %lfs\n", t2 - t1);
    }

    MPI_Finalize();
    return 0;
}