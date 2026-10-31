#include <iostream>
#include <omp.h>

void run_matrix_addition(int num_threads) {
    const int ROWS = 3;
    const int COLS = 3;

    int A[ROWS][COLS] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    int B[ROWS][COLS] = {
        {9, 8, 7},
        {6, 5, 4},
        {3, 2, 1}
    };
    int C[ROWS][COLS] = {0};

    omp_set_num_threads(num_threads);

    std::cout << "\n====================================\n";
    std::cout << "  RUNNING WITH TARGET THREADS: " << num_threads << "\n";
    std::cout << "====================================\n";

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();

        #pragma omp critical
        {
            std::cout << "[INFO] Thread ID: " << thread_id 
                      << " | Total Active Threads: " << total_threads << "\n";
        }

        #pragma omp for
        for (int i = 0; i < ROWS; ++i) {
            for (int j = 0; j < COLS; ++j) {
                C[i][j] = A[i][j] + B[i][j];
            }
        }
    }

    std::cout << "\nResultant Matrix C (A + B):\n";
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            std::cout << C[i][j] << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    run_matrix_addition(2);
    run_matrix_addition(4);
    run_matrix_addition(8);

    return 0;
}

