/*
 * DOT PRODUCT - OPENMP
 *
 * Compile: gcc -fopenmp dot_openmp.c -o dot_openmp
 * Run:     ./dot_openmp
 * Optional: OMP_NUM_THREADS=4 ./dot_openmp
 */

#include <stdio.h>
#include <omp.h>

int main(void) {
    int A[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int B[] = {2, 3, 4, 5, 6, 7, 8, 9};
    int n = sizeof(A) / sizeof(A[0]);
    int dot = 0;

    #pragma omp parallel for reduction(+:dot)
    for (int i = 0; i < n; i++) {
        dot += A[i] * B[i];
    }

    printf("Dot Product = %d\\n", dot);
    return 0;
}
