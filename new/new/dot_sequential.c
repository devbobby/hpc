/*
 * DOT PRODUCT - SEQUENTIAL C
 *
 * Compile: gcc dot_sequential.c -o dot_sequential
 * Run:     ./dot_sequential
 */

#include <stdio.h>

int main(void) {
    int A[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int B[] = {2, 3, 4, 5, 6, 7, 8, 9};
    int n = sizeof(A) / sizeof(A[0]);
    int dot = 0;

    for (int i = 0; i < n; i++) {
        dot += A[i] * B[i];
    }

    printf("Dot Product = %d\\n", dot);
    return 0;
}
