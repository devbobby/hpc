#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <chrono>
using namespace std;
using Clock=chrono::steady_clock;

int main(int argc,char** argv) {
    int n=(argc>1)?atoi(argv[1]):1000;
    if(n<1){cerr<<"Size must be positive\n";return 1;}
    vector<double> I((size_t)n*n), O((size_t)n*n,0.0);
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) I[(size_t)i*n+j]=((i*13+j*5)%251)/251.0;
    const double K[3][3]={{1,1,1},{1,1,1},{1,1,1}};
    auto start=Clock::now();
    for(int i=1;i<n-1;i++) for(int j=1;j<n-1;j++) {
        double v=0;
        for(int u=-1;u<=1;u++) for(int w=-1;w<=1;w++)
            v+=I[(size_t)(i+u)*n+(j+w)]*K[u+1][w+1];
        O[(size_t)i*n+j]=v;
    }
    double seconds=chrono::duration<double>(Clock::now()-start).count();
    double sum=0;for(double x:O)sum+=x;
    cout<<fixed<<setprecision(6)<<"Application=convolution Size="<<n<<" Model=Sequential Workers=1 Time_s="
        <<seconds<<" Checksum="<<setprecision(10)<<sum<<"\n";
}
