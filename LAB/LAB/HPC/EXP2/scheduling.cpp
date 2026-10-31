#include <iostream>
#include <omp.h>

void run_scheduling_test(int SIZE, int num_threads, int type) {
    double** A = new double*[SIZE];
    double** B = new double*[SIZE];
    double** C = new double*[SIZE];
    
    for (int i = 0; i < SIZE; ++i) {
        A[i] = new double[SIZE];
        B[i] = new double[SIZE];
        C[i] = new double[SIZE]();
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
            C[i][j] = sum;
        }
    }
    double end_serial = omp_get_wtime();
    double time_serial = end_serial - start_serial;

    omp_set_num_threads(num_threads);
    double start_parallel = 0.0, end_parallel = 0.0;

    if (type == 1) {
        start_parallel = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                double sum = 0.0;
                for (int k = 0; k < SIZE; ++k) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        end_parallel = omp_get_wtime();
        std::cout << "Static    | ";
    } 
    else if (type == 2) {
        start_parallel = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                double sum = 0.0;
                for (int k = 0; k < SIZE; ++k) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        end_parallel = omp_get_wtime();
        std::cout << "Dynamic   | ";
    } 
    else if (type == 3) {
        start_parallel = omp_get_wtime();
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                double sum = 0.0;
                for (int k = 0; k < SIZE; ++k) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        end_parallel = omp_get_wtime();
        std::cout << "Guided    | ";
    }

    double time_parallel = end_parallel - start_parallel;
    double speedup = time_serial / time_parallel;

    std::cout << time_parallel << " seconds | " << speedup << "x\n";

    for (int i = 0; i < SIZE; ++i) {
        delete[] A[i];
        delete[] B[i];
        delete[] C[i];
    }
    delete[] A;
    delete[] B;
    delete[] C;
}

int main() {
    int max_threads = omp_get_max_threads();
    const int MATRIX_SIZE = 1000;

    std::cout << "========================================================\n";
    std::cout << "BENCHMARKING OPENMP SCHEDULING (Threads: " << max_threads << ")\n";
    std::cout << "========================================================\n";
    std::cout << "Scheduling Type | Execution Time | Speedup\n";
    std::cout << "--------------------------------------------------------\n";

    run_scheduling_test(MATRIX_SIZE, max_threads, 1);
    run_scheduling_test(MATRIX_SIZE, max_threads, 2);
    run_scheduling_test(MATRIX_SIZE, max_threads, 3);

    std::cout << "========================================================\n";
    return 0;
}

