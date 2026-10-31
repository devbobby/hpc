#include <iostream>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <cmath>
#include <algorithm>

using namespace std;

void initializeMatrix(vector<vector<double>>& A, int N, double value)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = value;
}

void matrixMultiplySequential(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C,
    int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double sum = 0.0;

            for (int k = 0; k < N; k++)
                sum += A[i][k] * B[k][j];

            C[i][j] = sum;
        }
    }
}

void matrixMultiplyBlocked(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    vector<vector<double>>& C,
    int N)
{
    const int BLOCK = 32;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < N; ii += BLOCK)
    {
        for (int jj = 0; jj < N; jj += BLOCK)
        {
            int iEnd = min(ii + BLOCK, N);
            int jEnd = min(jj + BLOCK, N);

            for (int i = ii; i < iEnd; i++)
            {
                for (int j = jj; j < jEnd; j++)
                {
                    double sum = 0.0;

                    for (int kk = 0; kk < N; kk += BLOCK)
                    {
                        int kEnd = min(kk + BLOCK, N);

                        for (int k = kk; k < kEnd; k++)
                        {
                            sum += A[i][k] * B[k][j];
                        }
                    }

                    C[i][j] = sum;
                }
            }
        }
    }
}

bool verify(
    const vector<vector<double>>& C1,
    const vector<vector<double>>& C2,
    int N)
{
    const double EPS = 1e-9;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (fabs(C1[i][j] - C2[i][j]) > EPS)
                return false;
        }
    }

    return true;
}

int main()
{
    int N;

    cout << "Enter matrix size: ";
    cin >> N;

    vector<vector<double>> A(N, vector<double>(N));
    vector<vector<double>> B(N, vector<double>(N));
    vector<vector<double>> Cseq(N, vector<double>(N, 0.0));
    vector<vector<double>> Cpar(N, vector<double>(N, 0.0));

    initializeMatrix(A, N, 1.0);
    initializeMatrix(B, N, 2.0);

    double start, end;

    start = omp_get_wtime();

    matrixMultiplySequential(A, B, Cseq, N);

    end = omp_get_wtime();

    double sequentialTime = end - start;

    start = omp_get_wtime();

    matrixMultiplyBlocked(A, B, Cpar, N);

    end = omp_get_wtime();

    double parallelTime = end - start;

    double speedup = sequentialTime / parallelTime;

    int threads = omp_get_max_threads();

    double efficiency =
        (speedup / threads) * 100.0;

    bool correct = verify(Cseq, Cpar, N);

    cout << fixed << setprecision(8);

    cout << "\nMatrix Size     : "
         << N << " x " << N << endl;

    cout << "Threads         : "
         << threads << endl;

    cout << "Sequential Time : "
         << sequentialTime << " seconds" << endl;

    cout << "Parallel Time   : "
         << parallelTime << " seconds" << endl;

    cout << "Speedup         : "
         << speedup << endl;

    cout << "Efficiency      : "
         << efficiency << "%" << endl;

    cout << "Correctness     : "
         << (correct ? "PASS" : "FAIL") << endl;

    return 0;
}
