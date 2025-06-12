#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
using namespace std;
using namespace chrono;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    
    double start_train = MPI_Wtime();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    double end_train = MPI_Wtime();
    time_train = end_train - start_train;

    q.init();
    cout << "here" << endl;
    int curr_num = 0;
    double start = MPI_Wtime();
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;
    // std::ofstream a("./output/results.txt");
    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();
        if (q.total_guesses - curr_num >= 100000)
        {
            curr_num = q.total_guesses;

            // 在此处更改实验生成的猜测上限
            int generate_n=1e7;
            if (history + q.total_guesses > generate_n)
            {
                double end = MPI_Wtime();
                time_guess = end - start;
                cout << "Guess time:" << time_guess - time_hash << "seconds"<< endl;
                cout << "Hash time:" << time_hash << "seconds"<<endl;
                cout << "Train time:" << time_train <<"seconds"<<endl;
                break;
            }
        }
        // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
        // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
        if (curr_num > 1000000)
        {
            double start_hash = MPI_Wtime();
            bit32 state[4];
            for (string pw : q.guesses)
            {
                // TODO：对于SIMD实验，将这里替换成你的SIMD MD5函数
                MD5Hash(pw, state);

                // 以下注释部分用于输出猜测和哈希，但是由于自动测试系统不太能写文件，所以这里你可以改成cout
                // a<<pw<<"\t";
                // for (int i1 = 0; i1 < 4; i1 += 1)
                // {
                //     a << std::setw(8) << std::setfill('0') << hex << state[i1];
                // }
                // a << endl;
            }
            // 在这里对哈希所需的总时长进行计算
            double end_hash = MPI_Wtime();
            time_hash += end_hash - start_hash;

            // 记录已经生成的口令总数
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
    cout<<endl<<"1"<<endl;
    cout<<"2"<<endl;
    return 0;
}