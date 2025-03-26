#include <iostream>
#include<ctime>
#include <stdlib.h>
#include <windows.h>
#include<vector>
using namespace std;

long long head, tail, freq;
const int N = 10000;
int a[N];
int sum[N];
int b[N][N];

void init() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            b[i][j] = i + j;  // 填充矩阵元素
        }
        a[i] = i;
    }
}

void normal(int n)//平凡算法
{
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < n; i++)
    {
        sum[i] = 0;
        for (int j = 0; j < n; j++)
        {
            sum[i] += b[j][i] * a[j];
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << (tail - head) * 1000.0 / freq << ",";
}
void normal1(int n)//平凡算法
{
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int k = 0; k < 100; k++) {
        for (int i = 0; i < n; i++)
        {
            sum[i] = 0;
            for (int j = 0; j < n; j++)
            {
                sum[i] += b[j][i] * a[j];
            }
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << (tail - head) * 10.0 / freq << ",";
}

void optimize(int n)//优化算法
{
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < n; i++)
        sum[i] = 0;
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            sum[i] += b[j][i] * a[j];
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << (tail - head) * 1000.0 / freq;
}
void optimize1(int n)//优化算法
{
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < 100; i++) {
        for (int i = 0; i < n; i++)
            sum[i] = 0;
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                sum[i] += b[j][i] * a[j];
            }
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << (tail - head) * 10.0 / freq;
}

int main()
{
    init();
    cout << "问题规模," << "平均算法用时(ms)," << "优化算法用时(ms)"<<endl;
    vector<int> test_sizes = {
    100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
    2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000
    };
    vector<int>small_size = { 10,20,30,40,50,60,70,80,90 };
    for (int n : small_size) {
        cout << n << ",";
        normal1(n);
        optimize1(n);
        cout << endl;
    }
    for (int n:test_sizes) {
        cout << n << ",";
        normal(n);
        optimize(n);
        cout << endl;
    }
    return 0;
}
