#include <iostream>
#include<ctime>
#include <stdlib.h>
#include <windows.h>
#include<vector>
using namespace std;

const int N =200000000 ;
double sum = 0;
double a[N];
long long head, tail, freq;
double sum0 = 0,sum1=0;


void init()
{
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
    }
}

void normal(int n)//平凡算法
{
    for (int i = 0; i < n; i++)
    {
        sum += a[i];
    }

}

void multiplelink(int n)//多路链式
{
    sum0 = 0;
    sum1 = 0;
    for (int i0 = 0, i1 = n / 2; i0 < n / 2 && i1 < n; i0++, i1++)
    {
        sum0 += a[i0];
        sum1 += a[i1];
    }
    sum = sum0 + sum1;
}

void recursion(int n)//递归
{
    if (n == 1)
    {
        return;
    }
    else
    {
        for (int i = 0; i < n / 2; i++)
            a[i] += a[n - i - 1];
        n = n / 2;
        recursion(n);
    }
}

void doubleloop(int n)//双重循环
{
    for (int i = n; i > 1; i = i / 2)
    {
        for (int j = 0; j < i / 2; j++)
        {
            a[j] = a[j*2]+a[j*2+1];
        }
    }
}

int main()
{
    long long head, tail, freq;
    cout<< "问题规模," << "普通算法用时(ms)," << "多链路式算法用时(ms)," << "递归算法用时(ms)," << "双重循环用时(ms)" << endl;

    init();
    int t = 10;
    vector<int> test_sizes = {
        // 小规模
        1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,10000,

        // 中规模
        100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000,1000000,

        //大规模
        1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000
    };
    for (int n:test_sizes) {
        cout << n << ",";
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for (int i = 0; i < t; i++) {
            normal(n);
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << (tail - head) * 1000.0 / freq/t << ",";
        sum = 0;
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for (int i = 0; i < t; i++) {
            multiplelink(n);
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << (tail - head) * 1000.0 / freq / t << ",";
        sum = 0;
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for (int i = 0; i < t; i++) {
            recursion(n);
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << (tail - head) * 1000.0 / freq / t << ",";
        sum = 0;
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for (int i = 0; i < t; i++) {
            doubleloop(n);
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << (tail - head) * 1000.0 / freq / t << endl;
    }
    return 0;
}
