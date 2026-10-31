#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>
#include <cmath>

using namespace std;

void initializeMatrix(vector<vector<double>>& M, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            M[i][j] = (i + j) % 100 + 1;
        }
    }
}

void sequentialMultiply(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C,
    int n
) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;

            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }

            C[i][j] = sum;
        }
    }
}

void parallelStatic(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C,
    int n
) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;

            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }

            C[i][j] = sum;
        }
    }
}

void parallelDynamic(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C,
    int n
) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;

            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }

            C[i][j] = sum;
        }
    }
}

bool verify(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    int n
) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(A[i][j] - B[i][j]) > 1e-6) {
                return false;
            }
        }
    }

    return true;
}

int main() {
    int n;

    cout << "Enter matrix size: ";
    cin >> n;

    vector<vector<double>> A(n, vector<double>(n));
    vector<vector<double>> B(n, vector<double>(n));

    vector<vector<double>> C_seq(n, vector<double>(n));
    vector<vector<double>> C_static(n, vector<double>(n));
    vector<vector<double>> C_dynamic(n, vector<double>(n));

    initializeMatrix(A, n);
    initializeMatrix(B, n);

    double start = omp_get_wtime();

    sequentialMultiply(A, B, C_seq, n);

    double seq_time = omp_get_wtime() - start;

    start = omp_get_wtime();

    parallelStatic(A, B, C_static, n);

    double static_time = omp_get_wtime() - start;

    start = omp_get_wtime();

    parallelDynamic(A, B, C_dynamic, n);

    double dynamic_time = omp_get_wtime() - start;

    int threads = omp_get_max_threads();

    cout << fixed << setprecision(8);

    cout << "\nMatrix Size     : " << n << " x " << n << "\n";
    cout << "Threads         : " << threads << "\n";

    cout << "\nSequential Time : " << seq_time << " seconds\n";
    cout << "Static Time     : " << static_time << " seconds\n";
    cout << "Dynamic Time    : " << dynamic_time << " seconds\n";

    cout << "\nStatic Speedup  : " << seq_time / static_time << "\n";
    cout << "Dynamic Speedup : " << seq_time / dynamic_time << "\n";

    cout << "\nStatic Correctness  : "
         << (verify(C_seq, C_static, n) ? "PASS" : "FAIL");

    cout << "\nDynamic Correctness : "
         << (verify(C_seq, C_dynamic, n) ? "PASS" : "FAIL");

    cout << "\n";

    return 0;
}
