#include <iostream>
using namespace  std;
#include <omp.h>

int main(){

    
    
    #pragma omp parallel
    {
    #pragma omp critical
    {
        cout << "hello this is my thread no. " << omp_get_thread_num() <<endl;
    }
      
    }
    
    return 0;
}

