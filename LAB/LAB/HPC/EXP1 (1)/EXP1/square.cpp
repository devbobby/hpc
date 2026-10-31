#include <iostream>
#include <omp.h>

int main() {
    const int N = 1000;
    long long parallel_results[N];
    long long serial_results[N];

    double start_parallel = omp_get_wtime();

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();

        #pragma omp critical
        {
            std::cout << "Thread ID: " << thread_id 
                      << " out of " << total_threads << " threads active\n";
        }

        #pragma omp for
        for (int i = 0; i < N; ++i) {
            parallel_results[i] = (long long)i * i;
        }
    }
    double end_parallel = omp_get_wtime();
    double time_parallel = end_parallel - start_parallel;

    double start_serial = omp_get_wtime();
    for (int i = 0; i < N; ++i) {
        serial_results[i] = (long long)i * i;
    }
    double end_serial = omp_get_wtime();
    double time_serial = end_serial - start_serial;

    double speedup = time_serial / time_parallel;

    std::cout << "\n--- Execution Results ---\n";
    std::cout << "Serial Time: " << time_serial << " seconds\n";
    std::cout << "Parallel Time: " << time_parallel << " seconds\n";
    std::cout << "Speedup: " << speedup << "x\n";

    return 0;
}

