#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <omp.h>

using namespace std;

void mergeArrays(vector<int>& a, int left, int mid, int right) {
    vector<int> temp;
    temp.reserve(right - left + 1);

    int i = left;
    int j = mid + 1;

    while (i <= mid && j <= right) {
        if (a[i] <= a[j])
            temp.push_back(a[i++]);
        else
            temp.push_back(a[j++]);
    }

    while (i <= mid)
        temp.push_back(a[i++]);

    while (j <= right)
        temp.push_back(a[j++]);

    for (int k = 0; k < (int)temp.size(); k++)
        a[left + k] = temp[k];
}

void sequentialMergeSort(vector<int>& a, int left, int right) {
    if (left >= right)
        return;

    int mid = left + (right - left) / 2;

    sequentialMergeSort(a, left, mid);
    sequentialMergeSort(a, mid + 1, right);

    mergeArrays(a, left, mid, right);
}

void parallelMergeSort(vector<int>& a, int left, int right, int depth) {
    if (left >= right)
        return;

    if (right - left < 1000) {
        sequentialMergeSort(a, left, right);
        return;
    }

    int mid = left + (right - left) / 2;

    #pragma omp task shared(a) if(depth < 4)
    parallelMergeSort(a, left, mid, depth + 1);

    #pragma omp task shared(a) if(depth < 4)
    parallelMergeSort(a, mid + 1, right, depth + 1);

    #pragma omp taskwait

    mergeArrays(a, left, mid, right);
}

int main() {
    int n;

    cout << "Enter array size: ";
    cin >> n;

    vector<int> original(n);

    mt19937 generator(42);
    uniform_int_distribution<int> distribution(1, 1000000);

    for (int i = 0; i < n; i++)
        original[i] = distribution(generator);

    vector<int> seq = original;
    vector<int> par = original;

    double start = omp_get_wtime();

    sequentialMergeSort(seq, 0, n - 1);

    double sequentialTime = omp_get_wtime() - start;

    start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            parallelMergeSort(par, 0, n - 1, 0);
        }
    }

    double parallelTime = omp_get_wtime() - start;

    bool correct = (seq == par);

    int threads = omp_get_max_threads();

    cout << "\nSequential Time : "
         << sequentialTime << " seconds\n";

    cout << "Parallel Time   : "
         << parallelTime << " seconds\n";

    cout << "Threads         : "
         << threads << "\n";

    cout << "Speedup         : "
         << sequentialTime / parallelTime << "\n";

    cout << "Correctness     : "
         << (correct ? "PASS" : "FAIL") << "\n";

    return 0;
}
