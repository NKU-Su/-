      
#include "PCFG_mpi.h"
#include <mpi.h>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

using namespace std;


void PriorityQueue::CalProb(PT &pt)
{
    // 计算一个PT本身的概率
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init()
{
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
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

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Generate(priority.front());
    // 只有主进程更新优先队列
    if (rank == 0)
    {
        vector<PT> new_pts = priority.front().NewPTs();
        for (PT pt : new_pts)
        {
            CalProb(pt);
            for (auto iter = priority.begin(); iter != priority.end(); iter++)
            {
                if (iter != priority.end() - 1 && iter != priority.begin())
                {
                    if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                    {
                        priority.emplace(iter + 1, pt);
                        break;
                    }
                }
                if (iter == priority.end() - 1)
                {
                    priority.emplace_back(pt);
                    break;
                }
                if (iter == priority.begin() && iter->prob < pt.prob)
                {
                    priority.emplace(iter, pt);
                    break;
                }
            }
        }
        priority.erase(priority.begin());
    }
}

vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        int init_pivot = pivot;

        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            curr_indices[i] += 1;

            if (curr_indices[i] < max_indices[i])
            {
                pivot = i;
                res.emplace_back(*this);
            }

            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}

void PriorityQueue::Generate(PT pt)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 只有rank 0计算概率
    if (rank == 0)
    {
        CalProb(pt);
    }

    // 广播PT基本信息
    int content_size = 0;
    if (rank == 0)
    {
        content_size = pt.content.size();
    }
    MPI_Bcast(&content_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 单segment处理
    if (content_size == 1)
    {
        // 需要从rank 0广播的数据
        int segment_type = 0;
        int segment_index = 0;
        int total_values = 0;

        if (rank == 0)
        {
            if (pt.content[0].type == 1)
            {
                segment_type = 1;
                segment_index = m.FindLetter(pt.content[0]);
            }
            else if (pt.content[0].type == 2)
            {
                segment_type = 2;
                segment_index = m.FindDigit(pt.content[0]);
            }
            else
            {
                segment_type = 3;
                segment_index = m.FindSymbol(pt.content[0]);
            }
            total_values = pt.max_indices[0];
        }

        // 广播必要信息
        int info[3] = {segment_type, segment_index, total_values};
        MPI_Bcast(info, 3, MPI_INT, 0, MPI_COMM_WORLD);
        segment_type = info[0];
        segment_index = info[1];
        total_values = info[2];

        // 计算每个进程处理的数据范围
        int items_per_proc = total_values / size;
        int remainder = total_values % size;
        int start_idx = rank * items_per_proc + min(rank, remainder);
        int end_idx = start_idx + items_per_proc + (rank < remainder ? 1 : 0);
        int local_count = end_idx - start_idx;

        // 创建本地结果数组
        vector<string> local_results(local_count);

        segment *a;
        if (segment_type == 1)
        {
            a = &m.letters[segment_index];
        }
        else if (segment_type == 2)
        {
            a = &m.digits[segment_index];
        }
        else
        {
            a = &m.symbols[segment_index];
        }

        // 每个进程生成自己负责的部分
        for (int i = 0; i < local_count; i++)
        {
            local_results[i] = a->ordered_values[start_idx + i];
        }

        // ===== 优化的通信部分 =====

        // 1. 准备接收计数和偏移数组
        int *recvcounts = new int[size];
        int *displs = new int[size];

        // 每个进程计算自己的字符串长度总和
        vector<int> str_lengths(local_count);
        int local_total_size = 0;

        for (int i = 0; i < local_count; i++)
        {
            str_lengths[i] = local_results[i].length();
            local_total_size += str_lengths[i];
        }

        // 2. 收集所有进程的字符串计数
        int *all_counts = nullptr;
        if (rank == 0)
        {
            all_counts = new int[size];
        }

        MPI_Gather(&local_count, 1, MPI_INT, all_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 3. 收集所有进程的字符串长度总和
        int *all_sizes = nullptr;
        if (rank == 0)
        {
            all_sizes = new int[size];
        }

        MPI_Gather(&local_total_size, 1, MPI_INT, all_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 4. 准备接收所有字符串长度
        vector<int> all_str_lengths;
        if (rank == 0)
        {
            // 设置接收计数和偏移
            int total_count = 0;
            for (int i = 0; i < size; i++)
            {
                recvcounts[i] = all_counts[i];
                displs[i] = total_count;
                total_count += all_counts[i];
            }

            all_str_lengths.resize(total_count);
        }

        // 5. 收集所有字符串长度
        MPI_Gatherv(str_lengths.data(), local_count, MPI_INT,
                    all_str_lengths.data(), recvcounts, displs, MPI_INT,
                    0, MPI_COMM_WORLD);

        // 6. 处理字符串内容
        if (rank == 0)
        {
            // 复制本地结果
            for (int i = 0; i < local_count; i++)
            {
                guesses.push_back(move(local_results[i]));
            }

            // 从所有进程收集字符串内容
            for (int src_rank = 1; src_rank < size; src_rank++)
            {
                int remote_count = all_counts[src_rank];

                if (remote_count <= 0)
                    continue;

                // 计算总数据大小
                int total_size = all_sizes[src_rank];

                // 接收字符串内容
                char *buffer = new char[total_size];
                MPI_Status status;
                MPI_Recv(buffer, total_size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &status);

                // 解包字符串
                int offset = 0;
                for (int i = 0; i < remote_count; i++)
                {
                    int str_len = all_str_lengths[displs[src_rank] + i];
                    guesses.push_back(string(buffer + offset, str_len));
                    offset += str_len;
                }

                delete[] buffer;
            }

            total_guesses += total_values;

            delete[] all_counts;
            delete[] all_sizes;
        }
        else if (local_count > 0)
        {
            // 非rank 0进程发送字符串内容
            char *buffer = new char[local_total_size];
            int offset = 0;

            for (int i = 0; i < local_count; i++)
            {
                memcpy(buffer + offset, local_results[i].c_str(), str_lengths[i]);
                offset += str_lengths[i];
            }

            // 发送字符串内容
            MPI_Send(buffer, local_total_size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);

            delete[] buffer;
        }

        delete[] recvcounts;
        delete[] displs;
    }
    else
    {
        // 多segment情况处理
        string prefix;
        int last_segment_type = 0;
        int last_segment_index = 0;
        int total_values = 0;
        vector<int> curr_indices;

        if (rank == 0)
        {
            // 构建前缀信息
            last_segment_type = pt.content[pt.content.size() - 1].type;
            if (last_segment_type == 1)
            {
                last_segment_index = m.FindLetter(pt.content[pt.content.size() - 1]);
            }
            else if (last_segment_type == 2)
            {
                last_segment_index = m.FindDigit(pt.content[pt.content.size() - 1]);
            }
            else
            {
                last_segment_index = m.FindSymbol(pt.content[pt.content.size() - 1]);
            }

            total_values = pt.max_indices[pt.content.size() - 1];
            curr_indices = pt.curr_indices;
        }

        // 广播当前索引数组大小
        int indices_size = 0;
        if (rank == 0)
        {
            indices_size = curr_indices.size();
        }
        MPI_Bcast(&indices_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 广播当前索引数组
        if (rank != 0)
        {
            curr_indices.resize(indices_size);
        }
        MPI_Bcast(curr_indices.data(), indices_size, MPI_INT, 0, MPI_COMM_WORLD);

        // 广播内容类型
        vector<int> content_types;
        if (rank == 0)
        {
            content_types.resize(pt.content.size());
            for (size_t i = 0; i < pt.content.size(); i++)
            {
                content_types[i] = pt.content[i].type;
            }
        }
        else
        {
            content_types.resize(content_size);
        }
        MPI_Bcast(content_types.data(), content_size, MPI_INT, 0, MPI_COMM_WORLD);

        // 广播其他必要信息
        int info[3] = {last_segment_type, last_segment_index, total_values};
        if (rank == 0)
        {
            info[0] = last_segment_type;
            info[1] = last_segment_index;
            info[2] = total_values;
        }
        MPI_Bcast(info, 3, MPI_INT, 0, MPI_COMM_WORLD);
        last_segment_type = info[0];
        last_segment_index = info[1];
        total_values = info[2];

        // 广播segment索引
        vector<int> segment_indices;
        if (rank == 0)
        {
            segment_indices.resize(pt.content.size());
            for (size_t i = 0; i < pt.content.size(); i++)
            {
                if (pt.content[i].type == 1)
                {
                    segment_indices[i] = m.FindLetter(pt.content[i]);
                }
                else if (pt.content[i].type == 2)
                {
                    segment_indices[i] = m.FindDigit(pt.content[i]);
                }
                else
                {
                    segment_indices[i] = m.FindSymbol(pt.content[i]);
                }
            }
        }
        else
        {
            segment_indices.resize(content_size);
        }
        MPI_Bcast(segment_indices.data(), content_size, MPI_INT, 0, MPI_COMM_WORLD);

        // 每个进程都构建相同的前缀
        prefix.reserve(100); // 预估的前缀长度
        int seg_idx = 0;
        for (int idx : curr_indices)
        {
            if (content_types[seg_idx] == 1)
            {
                prefix += m.letters[segment_indices[seg_idx]].ordered_values[idx];
            }
            else if (content_types[seg_idx] == 2)
            {
                prefix += m.digits[segment_indices[seg_idx]].ordered_values[idx];
            }
            else if (content_types[seg_idx] == 3)
            {
                prefix += m.symbols[segment_indices[seg_idx]].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == content_size - 1)
            {
                break;
            }
        }

        // 计算每个进程处理的数据范围
        int items_per_proc = total_values / size;
        int remainder = total_values % size;
        int start_idx = rank * items_per_proc + min(rank, remainder);
        int end_idx = start_idx + items_per_proc + (rank < remainder ? 1 : 0);
        int local_count = end_idx - start_idx;

        // 创建本地结果数组
        vector<string> local_results(local_count);

        // 获取最后一个segment
        segment *a;
        if (last_segment_type == 1)
        {
            a = &m.letters[last_segment_index];
        }
        else if (last_segment_type == 2)
        {
            a = &m.digits[last_segment_index];
        }
        else
        {
            a = &m.symbols[last_segment_index];
        }

        // 每个进程生成自己负责的部分
        for (int i = 0; i < local_count; i++)
        {
            local_results[i] = prefix + a->ordered_values[start_idx + i];
        }

        // ===== 优化的通信部分（与单segment相同）=====

        // 1. 准备接收计数和偏移数组
        int *recvcounts = new int[size];
        int *displs = new int[size];

        // 每个进程计算自己的字符串长度总和
        vector<int> str_lengths(local_count);
        int local_total_size = 0;

        for (int i = 0; i < local_count; i++)
        {
            str_lengths[i] = local_results[i].length();
            local_total_size += str_lengths[i];
        }

        // 2. 收集所有进程的字符串计数
        int *all_counts = nullptr;
        if (rank == 0)
        {
            all_counts = new int[size];
        }

        MPI_Gather(&local_count, 1, MPI_INT, all_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 3. 收集所有进程的字符串长度总和
        int *all_sizes = nullptr;
        if (rank == 0)
        {
            all_sizes = new int[size];
        }

        MPI_Gather(&local_total_size, 1, MPI_INT, all_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 4. 准备接收所有字符串长度
        vector<int> all_str_lengths;
        if (rank == 0)
        {
            // 设置接收计数和偏移
            int total_count = 0;
            for (int i = 0; i < size; i++)
            {
                recvcounts[i] = all_counts[i];
                displs[i] = total_count;
                total_count += all_counts[i];
            }

            all_str_lengths.resize(total_count);
        }

        // 5. 收集所有字符串长度
        MPI_Gatherv(str_lengths.data(), local_count, MPI_INT,
                    all_str_lengths.data(), recvcounts, displs, MPI_INT,
                    0, MPI_COMM_WORLD);

        // 6. 处理字符串内容
        if (rank == 0)
        {
            // 复制本地结果
            for (int i = 0; i < local_count; i++)
            {
                guesses.push_back(move(local_results[i]));
            }

            // 从所有进程收集字符串内容
            for (int src_rank = 1; src_rank < size; src_rank++)
            {
                int remote_count = all_counts[src_rank];

                if (remote_count <= 0)
                    continue;

                // 计算总数据大小
                int total_size = all_sizes[src_rank];

                // 接收字符串内容
                char *buffer = new char[total_size];
                MPI_Status status;
                MPI_Recv(buffer, total_size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &status);

                // 解包字符串
                int offset = 0;
                for (int i = 0; i < remote_count; i++)
                {
                    int str_len = all_str_lengths[displs[src_rank] + i];
                    guesses.push_back(string(buffer + offset, str_len));
                    offset += str_len;
                }

                delete[] buffer;
            }

            total_guesses += total_values;

            delete[] all_counts;
            delete[] all_sizes;
        }
        else if (local_count > 0)
        {
            // 非rank 0进程发送字符串内容
            char *buffer = new char[local_total_size];
            int offset = 0;

            for (int i = 0; i < local_count; i++)
            {
                memcpy(buffer + offset, local_results[i].c_str(), str_lengths[i]);
                offset += str_lengths[i];
            }

            // 发送字符串内容
            MPI_Send(buffer, local_total_size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);

            delete[] buffer;
        }

        delete[] recvcounts;
        delete[] displs;
    }
}





    