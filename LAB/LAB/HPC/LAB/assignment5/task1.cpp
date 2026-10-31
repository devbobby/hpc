#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <iomanip>
#include <climits>

using namespace std;

int main() {
    int N;

    cout << "Enter array size: ";
    cin >> N;

    double *A = new double[N];

    for (int i = 0; i < N; i++)
        A[i] = (rand() % 10000) / 100.0;

    double sum_seq = 0;
    double max_seq = A[0];
    double min_seq = A[0];

    double start = omp_get_wtime();

    for (int i = 0; i < N; i++)
        sum_seq += A[i];

    for (int i = 1; i < N; i++)
        if (A[i] > max_seq)
            max_seq = A[i];

    for (int i = 1; i < N; i++)
        if (A[i] < min_seq)
            min_seq = A[i];

    double avg_seq = sum_seq / N;

    double seq_time = omp_get_wtime() - start;

    double sum_par = 0;
    double max_par = A[0];
    double min_par = A[0];
    double avg_par = 0;

    start = omp_get_wtime();

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            double local_sum = 0;

            for (int i = 0; i < N; i++)
                local_sum += A[i];

            sum_par = local_sum;
        }

        #pragma omp section
        {
            double local_max = A[0];

            for (int i = 1; i < N; i++)
                if (A[i] > local_max)
                    local_max = A[i];

            max_par = local_max;
        }

        #pragma omp section
        {
            double local_min = A[0];

            for (int i = 1; i < N; i++)
                if (A[i] < local_min)
                    local_min = A[i];

            min_par = local_min;
        }

        #pragma omp section
        {
            double local_sum = 0;

            for (int i = 0; i < N; i++)
                local_sum += A[i];

            avg_par = local_sum / N;
        }
    }

    double par_time = omp_get_wtime() - start;

    cout << fixed << setprecision(6);

    cout << "\nSequential Results\n";
    cout << "Sum     : " << sum_seq << endl;
    cout << "Maximum : " << max_seq << endl;
    cout << "Minimum : " << min_seq << endl;
    cout << "Average : " << avg_seq << endl;

    cout << "\nParallel Results\n";
    cout << "Sum     : " << sum_par << endl;
    cout << "Maximum : " << max_par << endl;
    cout << "Minimum : " << min_par << endl;
    cout << "Average : " << avg_par << endl;

    double speedup = seq_time / par_time;

    cout << "\nSequential Time : " << seq_time << " seconds";
    cout << "\nSections Time   : " << par_time << " seconds";
    cout << "\nSpeedup         : " << speedup << endl;

    if (abs(sum_seq - sum_par) < 1e-6 &&
        abs(max_seq - max_par) < 1e-6 &&
        abs(min_seq - min_par) < 1e-6 &&
        abs(avg_seq - avg_par) < 1e-6)
        cout << "Correctness     : PASS\n";
    else
        cout << "Correctness     : FAIL\n";

    delete[] A;

    return 0;
}
