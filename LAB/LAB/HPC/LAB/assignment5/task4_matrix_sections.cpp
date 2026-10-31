#include <iostream>
#include <omp.h>
#include <iomanip>

using namespace std;

void initialize(double **A, double **B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (i + j) % 100;
            B[i][j] = (i * j) % 100;
        }
    }
}

void addition(double **A, double **B, double **C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i][j] = A[i][j] + B[i][j];
}

void subtraction(double **A, double **B, double **C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            C[i][j] = A[i][j] - B[i][j];
}

void transpose(double **A, double **T, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            T[j][i] = A[i][j];
}

int main() {
    int sizes[] = {500, 1000, 2000};

    for (int s = 0; s < 3; s++) {

        int N = sizes[s];

        double **A = new double*[N];
        double **B = new double*[N];
        double **C1 = new double*[N];
        double **C2 = new double*[N];
        double **TA = new double*[N];
        double **TB = new double*[N];

        for (int i = 0; i < N; i++) {
            A[i] = new double[N];
            B[i] = new double[N];
            C1[i] = new double[N];
            C2[i] = new double[N];
            TA[i] = new double[N];
            TB[i] = new double[N];
        }

        initialize(A, B, N);

        double start = omp_get_wtime();

        addition(A, B, C1, N);
        subtraction(A, B, C2, N);
        transpose(A, TA, N);
        transpose(B, TB, N);

        double seq_time = omp_get_wtime() - start;

        start = omp_get_wtime();

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                addition(A, B, C1, N);
            }

            #pragma omp section
            {
                subtraction(A, B, C2, N);
            }

            #pragma omp section
            {
                transpose(A, TA, N);
            }

            #pragma omp section
            {
                transpose(B, TB, N);
            }
        }

        double par_time = omp_get_wtime() - start;

        cout << fixed << setprecision(6);

        cout << "\nMatrix Size : "
             << N << " x " << N << endl;

        cout << "Sequential Time : "
             << seq_time << " seconds\n";

        cout << "Sections Time   : "
             << par_time << " seconds\n";

        cout << "Speedup         : "
             << seq_time / par_time << endl;

        for (int i = 0; i < N; i++) {
            delete[] A[i];
            delete[] B[i];
            delete[] C1[i];
            delete[] C2[i];
            delete[] TA[i];
            delete[] TB[i];
        }

        delete[] A;
        delete[] B;
        delete[] C1;
        delete[] C2;
        delete[] TA;
        delete[] TB;
    }

    return 0;
}
