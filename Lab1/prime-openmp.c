#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define MAX_THREADS 8

int is_prime(int num) {
    for(int i = 2; i <= sqrt(num); i++) {
        if(num % i == 0) return 0;
    }
    return 1;
}

int main (int argc, const char *argv[]) {
    int n = atoi(argv[1]);
    double start, end;

    for (int j = 1; j <= MAX_THREADS; j *= 2) {
        omp_set_num_threads(j);

        int num = 0;
        double start = omp_get_wtime();

        #pragma omp parallel for reduction(+:num)
        for (int i = 2; i <= n; i++) {
            if(is_prime(i)) 
                num += 1;
        }

        end = omp_get_wtime();
        printf("Running on %d threads: Num = %d Time = %lf\n", j, num, end - start);
    }
}
