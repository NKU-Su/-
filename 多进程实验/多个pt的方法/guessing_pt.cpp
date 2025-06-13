#include "PCFG_pt.h"
#include <mpi.h>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

using namespace std;

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext(int batch_pt_num) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Step 1: Rank 0 determines the actual batch size to process
    int actual_batch = 0;
    vector<PT> batch_pts;
    
    if (rank == 0) {
        if (!priority.empty()) {
            actual_batch = min(batch_pt_num, static_cast<int>(priority.size()));
            if (actual_batch > 0) {
                batch_pts.assign(priority.begin(), priority.begin() + actual_batch);
            }
        }
    }

    // Broadcast the batch size to all processes
    MPI_Bcast(&actual_batch, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (actual_batch <= 0) return;

    // Step 2: Distribute work among processes using round-robin assignment
    vector<int> my_pt_indices;
    for (int i = rank; i < actual_batch; i += size) {
        my_pt_indices.push_back(i);
    }

    // Step 3: Process assigned PTs and generate guesses
    const int old_guesses_size = guesses.size();
    
    for (int pt_idx : my_pt_indices) {
        if (pt_idx < priority.size()) {
            Generate(priority[pt_idx]);
        }
    }

    // Step 4: Gather results to rank 0
    if (rank == 0) {
        // Handle rank 0's own results
        int my_new_guesses = guesses.size() - old_guesses_size;
        
        // Receive results from other ranks
        for (int src_rank = 1; src_rank < size; ++src_rank) {
            int remote_new_guesses = 0;
            MPI_Recv(&remote_new_guesses, 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (remote_new_guesses > 0) {
                // Receive string lengths
                vector<int> str_lengths(remote_new_guesses);
                MPI_Recv(str_lengths.data(), remote_new_guesses, MPI_INT, 
                         src_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Calculate total characters needed
                int total_chars = accumulate(str_lengths.begin(), str_lengths.end(), 0);
                
                // Receive all string data
                vector<char> buffer(total_chars);
                MPI_Recv(buffer.data(), total_chars, MPI_CHAR, 
                         src_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Unpack strings into guesses
                int buffer_pos = 0;
                for (int len : str_lengths) {
                    guesses.emplace_back(&buffer[buffer_pos], len);
                    buffer_pos += len;
                }
            }
        }

        total_guesses = guesses.size();

        // Step 5: Generate new PTs from processed batch
        vector<PT> new_pts;
        for (auto& pt : batch_pts) {
            vector<PT> pts = pt.NewPTs();
            for (PT& npt : pts) {
                CalProb(npt);
                new_pts.emplace_back(move(npt));
            }
        }

        // Remove processed PTs from priority queue
        priority.erase(priority.begin(), priority.begin() + actual_batch);

        // Insert new PTs maintaining probability order
        for (PT& pt : new_pts) {
            bool inserted = false;
            
            // Special case: insert at beginning if higher than first element
            if (!priority.empty() && pt.prob > priority.front().prob) {
                priority.insert(priority.begin(), pt);
                inserted = true;
            }
            
            // Find insertion point in the middle
            if (!inserted) {
                for (auto it = priority.begin(); it != priority.end(); ++it) {
                    if (next(it) == priority.end() || 
                        (pt.prob <= it->prob && pt.prob > next(it)->prob)) {
                        priority.insert(next(it), pt);
                        inserted = true;
                        break;
                    }
                }
            }
            
            // Insert at end if not inserted elsewhere
            if (!inserted) {
                priority.push_back(pt);
            }
        }
    } 
    else {
        // Non-rank 0 processes send their results
        int my_new_guesses = guesses.size() - old_guesses_size;
        MPI_Send(&my_new_guesses, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        if (my_new_guesses > 0) {
            // Prepare string lengths
            vector<int> str_lengths;
            for (int i = old_guesses_size; i < guesses.size(); ++i) {
                str_lengths.push_back(guesses[i].length());
            }

            // Send string lengths
            MPI_Send(str_lengths.data(), my_new_guesses, MPI_INT, 0, 1, MPI_COMM_WORLD);

            // Concatenate all strings into a single buffer
            vector<char> buffer;
            for (int i = old_guesses_size; i < guesses.size(); ++i) {
                const string& str = guesses[i];
                buffer.insert(buffer.end(), str.begin(), str.end());
            }

            // Send string data
            MPI_Send(buffer.data(), buffer.size(), MPI_CHAR, 0, 2, MPI_COMM_WORLD);
        }

        // Clean up local guesses
        guesses.resize(old_guesses_size);
        total_guesses = old_guesses_size;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}

// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现*****
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的，线程数是从1-8，调整使用的总线程数，并探索加速比随线程数的变化
        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            // cout << guess << endl;

            // 多线程编程这里需要多个guesses的vector去存不同线程的猜测结果
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }

        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的，线程数是从1-8，调整使用的总线程数，并探索加速比随线程数的变化
        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            // cout << temp << endl;

            // 多线程编程这里需要多个guesses的vector去存不同线程的猜测结果
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
    }
}