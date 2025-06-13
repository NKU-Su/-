      
#include "PCFG_PT.h"
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <vector>
#include <unordered_set>
#include <mpi.h>

using namespace std;



int main(int argc, char *argv[])
{
    // 初始化MPI环境
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;

    // 所有进程同步训练模型
    MPI_Barrier(MPI_COMM_WORLD);
    double start_train = MPI_Wtime();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    MPI_Barrier(MPI_COMM_WORLD);
    double end_train = MPI_Wtime();
    time_train = end_train - start_train;

    q.init();
    if (rank == 0)
    {
        cout << "here" << endl;
    }

    int curr_num = 0;
    // 同步所有进程，确保一起开始口令生成
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    long int history = 0;
    const long int generate_n = 1e7;

    bool run_flag = true;
    while (run_flag)
    {
        // 检查退出条件 1: 是否已达生成上限
        int limit_reached = 0;
        if (rank == 0)
        {
            limit_reached = (history + q.total_guesses >= generate_n) ? 1 : 0;
        }

        // 广播退出状态
        MPI_Bcast(&limit_reached, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (limit_reached == 1)
        {
            run_flag = false;
            break;
        }

        // 检查退出条件 2: 队列是否为空
        int queue_empty = 0;
        if (rank == 0)
        {
            queue_empty = q.priority.empty() ? 1 : 0;
        }

        // 广播队列状态
        MPI_Bcast(&queue_empty, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (queue_empty == 1)
        {
            run_flag = false;
            break;
        }

        // 调用口令生成函数（所有进程都参与）
        q.PopNext(8);

        // 只有rank 0更新计数器和处理哈希
        if (rank == 0)
        {
            q.total_guesses = q.guesses.size();
            // 为了避免内存超限，定期进行哈希处理
            if (q.total_guesses - curr_num >= 100000)
            {
                curr_num = q.total_guesses;
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

                double end_hash = MPI_Wtime();
                time_hash += end_hash - start_hash;

                // 记录已经生成的口令总数
                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
            }
        }

        // 每轮迭代结束，确保所有进程同步
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // 确保所有进程同步退出
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    // 只有rank 0输出统计信息
    if (rank == 0)
    {
        time_guess = end - start;
        cout << "Number of processes: " << size << endl;
        cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
        cout << "Hash time: " << time_hash << " seconds" << endl;
        cout << "Train time: " << time_train << " seconds" << endl;
        cout.flush();
    }

    MPI_Finalize();
    return 0;
}

    