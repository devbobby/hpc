#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <iomanip>

using namespace std;

void merge(int *A, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = new int[n1];
    int *R = new int[n2];

    for (int i = 0; i < n1; i++)
        L[i] = A[left + i];

    for (int i = 0; i < n2; i++)
        R[i] = A[mid + 1 + i];

    int i = 0;
    int j = 0;
    int k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j])
            A[k++] = L[i++];
        else
            A[k++] = R[j++];
    }

    while (i < n1)
        A[k++] = L[i++];

    while (j < n2)
        A[k++] = R[j++];

    delete[] L;
    delete[] R;
}

void mergeSortSequential(int *A, int left, int right) {
    if (left >= right)
        return;

    int mid = left + (right - left) / 2;

    mergeSortSequential(A, left, mid);
    mergeSortSequential(A, mid + 1, right);

    merge(A, left, mid, right);
}

void mergeSortParallel(int *A, int left, int right, int threshold) {
    if (left >= right)
        return;

    if (right - left + 1 <= threshold) {
        mergeSortSequential(A, left, right);
        return;
    }

    int mid = left + (right - left) / 2;

    #pragma omp task shared(A)
    mergeSortParallel(A, left, mid, threshold);

    #pragma omp task shared(A)
    mergeSortParallel(A, mid + 1, right, threshold);

    #pragma omp taskwait

    merge(A, left, mid, right);
}

bool checkSorted(int *A, int N) {
    for (int i = 1; i < N; i++)
        if (A[i - 1] > A[i])
            return false;

    return true;
}

int main() {
    int N;
    int threshold;

    cout << "Enter array size: ";
    cin >> N;

    cout << "Enter task threshold: ";
    cin >> threshold;

    int *A = new int[N];
    int *B = new int[N];

    srand(10);

    for (int i = 0; i < N; i++) {
        A[i] = rand();
        B[i] = A[i];
    }

    double start = omp_get_wtime();

    mergeSortSequential(A, 0, N - 1);

    double seq_time = omp_get_wtime() - start;

    start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            mergeSortParallel(B, 0, N - 1, threshold);
        }
    }

    double par_time = omp_get_wtime() - start;

    cout << fixed << setprecision(6);

    cout << "\nSequential Time : "
         << seq_time << " seconds\n";

    cout << "Parallel Time   : "
         << par_time << " seconds\n";

    cout << "Speedup         : "
         << seq_time / par_time << endl;

    cout << "Correctness     : "
         << (checkSorted(B, N) ? "PASS" : "FAIL")
         << endl;

    delete[] A;
    delete[] B;

    return 0;
}
