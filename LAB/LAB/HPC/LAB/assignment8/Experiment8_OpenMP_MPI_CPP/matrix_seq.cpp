#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <chrono>
using namespace std;
using Clock = chrono::steady_clock;

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 500;
    if (n < 1) { cerr << "Size must be positive\n"; return 1; }
    vector<double> A((size_t)n*n), B((size_t)n*n), C((size_t)n*n, 0.0);
    for (int i=0;i<n;i++) for (int j=0;j<n;j++) {
        A[(size_t)i*n+j] = ((i*17 + j*3) % 101) / 101.0;
        B[(size_t)i*n+j] = ((i*7 + j*11) % 97) / 97.0;
    }
    auto start = Clock::now();
    for (int i=0;i<n;i++)
        for (int k=0;k<n;k++) {
            double a = A[(size_t)i*n+k];
            for (int j=0;j<n;j++)
                C[(size_t)i*n+j] += a * B[(size_t)k*n+j];
        }
    double seconds = chrono::duration<double>(Clock::now()-start).count();
    cout << fixed << setprecision(6)
         << "Application=matrix Size=" << n << " Model=Sequential Workers=1 Time_s=" << seconds
         << " Checksum=";
    double sum=0; for (double x:C) sum += x;
    cout << setprecision(10) << sum << "\n";
}
