#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define MAX_THREADS 8

int main (int argc, const char *argv[]) {
    int n = atoi(argv[1]);
    double pi = 0.0;
    double start, end;

    for (int j = 1; j <= MAX_THREADS; j *= 2) {
        omp_set_num_threads(j);

        pi = 0.0;
        double start = omp_get_wtime();

        #pragma omp parallel for reduction(+:pi)
        for (int i = 0; i < n; i++) {
            if(i % 2 == 0)
                pi += (double)4 / (double)(2*i+1);
            else
                pi -= (double)4 / (double)(2*i+1);
        }

        end = omp_get_wtime();
        printf("Running on %d threads: PI = %f Time = %lf\n", j, pi, end - start);
    }
}
