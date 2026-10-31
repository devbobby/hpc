#include <iostream>
#include <omp.h>
#include <unistd.h>

using namespace std;

int main() {

    int A = 0;
    int B = 0;
    int C = 0;
    int D = 0;

    #pragma omp parallel
    {
        #pragma omp single
        {

            #pragma omp task depend(out:A)
            {
                cout << "Task A executing\n";
                A = 10;
            }

            #pragma omp task depend(in:A) depend(out:B)
            {
                cout << "Task B executing\n";
                B = A + 5;
            }

            #pragma omp task depend(in:A) depend(out:C)
            {
                cout << "Task C executing\n";
                C = A * 2;
            }

            #pragma omp task depend(in:B,C) depend(out:D)
            {
                cout << "Task D executing\n";
                D = B + C;
            }
        }
    }

    cout << "\nA = " << A << endl;
    cout << "B = " << B << endl;
    cout << "C = " << C << endl;
    cout << "D = " << D << endl;

    return 0;
}

