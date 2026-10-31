#include <iostream>
#include <omp.h>

void run_optimized_experiment(int SIZE, int num_threads) {
    double** A = new double*[SIZE];
    double** B = new double*[SIZE];
    double** C_serial = new double*[SIZE];
    double** C_parallel = new double*[SIZE];
    
    for (int i = 0; i < SIZE; ++i) {
        A[i] = new double[SIZE];
        B[i] = new double[SIZE];
        C_serial[i] = new double[SIZE]();
        C_parallel[i] = new double[SIZE]();
    }

    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            A[i][j] = 1.0;
            B[i][j] = 2.0;
        }
    }

   
    double start_serial = omp_get_wtime();
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            double sum = 0.0;
            for (int k = 0; k < SIZE; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C_serial[i][j] = sum;
        }
    }
    double end_serial = omp_get_wtime();
    double time_serial = end_serial - start_serial;

   
    omp_set_num_threads(num_threads);
    double start_parallel = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            double sum = 0.0;
            for (int k = 0; k < SIZE; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C_parallel[i][j] = sum;
        }
    }
    double end_parallel = omp_get_wtime();
    double time_parallel = end_parallel - start_parallel;

    double speedup = time_serial / time_parallel;
    double efficiency = speedup / num_threads;

    std::cout << num_threads << "\t|\t" 
              << time_parallel << "s\t|\t" 
              << speedup << "x\t|\t" 
              << efficiency * 100 << "%\n";

   
}

int main() {
    const int MATRIX_SIZE = 1000;
    int max_threads = omp_get_max_threads();

    std::cout << "=================== PERFORMANCE REPORT ===================\n";
    std::cout << "Threads\t|\tExec Time\t|\tSpeedup\t|\tEfficiency\n";
    std::cout << "----------------------------------------------------------\n";

    run_optimized_experiment(MATRIX_SIZE, 1);
    run_optimized_experiment(MATRIX_SIZE, 2);
    run_optimized_experiment(MATRIX_SIZE, 4);
    run_optimized_experiment(MATRIX_SIZE, 8);
    run_optimized_experiment(MATRIX_SIZE, max_threads);

    std::cout << "==========================================================\n";
    return 0;
}

