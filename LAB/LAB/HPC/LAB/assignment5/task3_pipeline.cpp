#include <iostream>
#include <omp.h>
#include <cmath>
#include <iomanip>

using namespace std;

void preprocessing(double *input, double *output, int N) {
    for (int i = 0; i < N; i++)
        output[i] = input[i] / 1000.0;
}

void computation(double *data, double *output, int N) {
    for (int i = 0; i < N; i++)
        output[i] = sqrt(data[i]) * data[i];
}

double postprocessing(double *data, int N) {
    double sum = 0;

    for (int i = 0; i < N; i++)
        sum += data[i];

    return sum / N;
}

int main() {
    int N;

    cout << "Enter data size: ";
    cin >> N;

    double *input = new double[N];
    double *processed = new double[N];
    double *computed = new double[N];

    for (int i = 0; i < N; i++)
        input[i] = (i % 1000) + 1;

    double sequential_result;

    double start = omp_get_wtime();

    preprocessing(input, processed, N);
    computation(processed, computed, N);
    sequential_result = postprocessing(computed, N);

    double seq_time = omp_get_wtime() - start;

    double parallel_result = 0;

    start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            {
                preprocessing(input, processed, N);
            }

            #pragma omp taskwait

            #pragma omp task
            {
                computation(processed, computed, N);
            }

            #pragma omp taskwait

            #pragma omp task
            {
                parallel_result = postprocessing(computed, N);
            }

            #pragma omp taskwait
        }
    }

    double task_time = omp_get_wtime() - start;

    cout << fixed << setprecision(8);

    cout << "\nSequential Result : "
         << sequential_result << endl;

    cout << "Parallel Result   : "
         << parallel_result << endl;

    cout << "\nSequential Time : "
         << seq_time << " seconds\n";

    cout << "Task Time       : "
         << task_time << " seconds\n";

    cout << "Speedup         : "
         << seq_time / task_time << endl;

    if (abs(sequential_result - parallel_result) < 1e-6)
        cout << "Correctness     : PASS\n";
    else
        cout << "Correctness     : FAIL\n";

    delete[] input;
    delete[] processed;
    delete[] computed;

    return 0;
}
