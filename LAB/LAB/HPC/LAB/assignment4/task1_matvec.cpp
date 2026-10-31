#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>
#include <cmath>

using namespace std;

int main() {
    int n;

    cout << "Enter matrix size: ";
    cin >> n;

    vector<vector<double>> A(n, vector<double>(n));
    vector<double> x(n);
    vector<double> y_seq(n);
    vector<double> y_par(n);

    for (int i = 0; i < n; i++) {
        x[i] = 1.0;

        for (int j = 0; j < n; j++) {
            A[i][j] = (i + j) % 100 + 1;
        }
    }

    double start = omp_get_wtime();

    for (int i = 0; i < n; i++) {
        double sum = 0.0;

        for (int j = 0; j < n; j++) {
            sum += A[i][j] * x[j];
        }

        y_seq[i] = sum;
    }

    double seq_time = omp_get_wtime() - start;

    start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double sum = 0.0;

        for (int j = 0; j < n; j++) {
            sum += A[i][j] * x[j];
        }

        y_par[i] = sum;
    }

    double par_time = omp_get_wtime() - start;

    bool correct = true;

    for (int i = 0; i < n; i++) {
        if (fabs(y_seq[i] - y_par[i]) > 1e-9) {
            correct = false;
            break;
        }
    }

    int threads = omp_get_max_threads();

    double speedup = seq_time / par_time;
    double efficiency = speedup / threads;

    cout << fixed << setprecision(8);

    cout << "\nSequential Time : " << seq_time << " seconds\n";
    cout << "Parallel Time   : " << par_time << " seconds\n";
    cout << "Threads         : " << threads << "\n";
    cout << "Speedup         : " << speedup << "\n";
    cout << "Efficiency      : " << efficiency * 100 << "%\n";
    cout << "Correctness     : " << (correct ? "PASS" : "FAIL") << "\n";

    return 0;
}
