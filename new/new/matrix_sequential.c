/*
 * MATRIX ADDITION - SEQUENTIAL C
 *
 * Compile: gcc matrix_sequential.c -o matrix_sequential
 * Run:     ./matrix_sequential
 */

#include <stdio.h>

#define N 3

int main(void) {
    int A[N][N] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    int B[N][N] = {
        {10, 20, 30},
        {40, 50, 60},
        {70, 80, 90}
    };

    int C[N][N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }

    printf("Result Matrix:\\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", C[i][j]);
        }
        printf("\\n");
    }

    return 0;
}
