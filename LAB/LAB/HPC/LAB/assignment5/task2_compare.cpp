#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <iomanip>

using namespace std;

int main() {
    int N;

    cout << "Enter array size: ";
    cin >> N;

    int *A = new int[N];

    for (int i = 0; i < N; i++)
        A[i] = rand() % 1000;

    long long sum1 = 0;
    int max1 = A[0];
    int min1 = A[0];
    long long even1 = 0;

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum1)
    for (int i = 0; i < N; i++)
        sum1 += A[i];

    #pragma omp parallel for reduction(max:max1)
    for (int i = 0; i < N; i++)
        max1 = max1 > A[i] ? max1 : A[i];

    #pragma omp parallel for reduction(min:min1)
    for (int i = 0; i < N; i++)
        min1 = min1 < A[i] ? min1 : A[i];

    #pragma omp parallel for reduction(+:even1)
    for (int i = 0; i < N; i++)
        if (A[i] % 2 == 0)
            even1++;

    double for_time = omp_get_wtime() - start;

    long long sum2 = 0;
    int max2 = A[0];
    int min2 = A[0];
    long long even2 = 0;

    start = omp_get_wtime();

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            long long local = 0;

            for (int i = 0; i < N; i++)
                local += A[i];

            sum2 = local;
        }

        #pragma omp section
        {
            int local = A[0];

            for (int i = 1; i < N; i++)
                if (A[i] > local)
                    local = A[i];

            max2 = local;
        }

        #pragma omp section
        {
            int local = A[0];

            for (int i = 1; i < N; i++)
                if (A[i] < local)
                    local = A[i];

            min2 = local;
        }

        #pragma omp section
        {
            long long local = 0;

            for (int i = 0; i < N; i++)
                if (A[i] % 2 == 0)
                    local++;

            even2 = local;
        }
    }

    double sections_time = omp_get_wtime() - start;

    cout << fixed << setprecision(6);

    cout << "\nResults\n";

    cout << "Sum     : " << sum1 << " / " << sum2 << endl;
    cout << "Maximum : " << max1 << " / " << max2 << endl;
    cout << "Minimum : " << min1 << " / " << min2 << endl;
    cout << "Even    : " << even1 << " / " << even2 << endl;

    cout << "\nParallel For Time : "
         << for_time << " seconds\n";

    cout << "Sections Time     : "
         << sections_time << " seconds\n";

    if (sum1 == sum2 &&
        max1 == max2 &&
        min1 == min2 &&
        even1 == even2)
        cout << "Correctness : PASS\n";
    else
        cout << "Correctness : FAIL\n";

    delete[] A;

    return 0;
}
