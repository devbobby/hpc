#include <iostream>
#include <omp.h>

void run_experiment(long long N, int num_threads) {
    long long* arr = new long long[N];
    for (long long i = 0; i < N; ++i) {
        arr[i] = i + 1;
    }

    double start_serial = omp_get_wtime();
    long long serial_sum = 0;
    for (long long i = 0; i < N; ++i) {
        serial_sum += arr[i];
    }
    double end_serial = omp_get_wtime();
    double time_serial = end_serial - start_serial;

    omp_set_num_threads(num_threads);
    double start_parallel = omp_get_wtime();
    long long parallel_sum = 0;

    #pragma omp parallel for
    for (long long i = 0; i < N; ++i) {
        #pragma omp critical
        {
            parallel_sum += arr[i];
        }
    }
    double end_parallel = omp_get_wtime();
    double time_parallel = end_parallel - start_parallel;

    double speedup = time_serial / time_parallel;

    std::cout << "------------------------------------------------\n";
    std::cout << "N                      : " << N << "\n";
    std::cout << "Threads Used           : " << num_threads << "\n";
    std::cout << "Serial Time            : " << time_serial << " seconds\n";
    std::cout << "Parallel Time          : " << time_parallel << " seconds\n";
    std::cout << "Speedup                : " << speedup << "x\n";


    delete[] arr;
}

int main() {
    int max_threads = omp_get_max_threads();

    std::cout << "================ EXPERIMENT REPORT ================\n";
    std::cout << "System Max Threads: " << max_threads << "\n\n";

    run_experiment(10000, max_threads);
    run_experiment(1000000, max_threads);
    run_experiment(10000000, max_threads);

    std::cout << "===================================================\n";
    return 0;
}

