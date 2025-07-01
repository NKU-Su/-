#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
using namespace std;

// ================== CUDA核函数（全局一维线程优化版，支持多猜测/线程） ==================
#define MAX_GUESS_LEN 128
#define MAX_BATCH 32
#define MAX_LAST_SEG 1024

__global__ void generate_guesses_kernel_flat(
    char* d_base,
    int* d_base_len,
    char* d_last_seg,
    int* d_last_seg_len,
    int* d_last_seg_offset,
    int* d_last_seg_count,
    int* d_pt_guess_offset, // 每个PT的猜测起始全局索引
    int total_guess_num,
    char* d_out
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;  // 总线程数

    // 每个线程循环处理多个猜测
    for (int idx = tid; idx < total_guess_num; idx += stride) {
        // 找到属于哪个PT
        int pt_idx = 0;
        while (pt_idx + 1 < MAX_BATCH && idx >= d_pt_guess_offset[pt_idx + 1]) pt_idx++;

        int local_guess_idx = idx - d_pt_guess_offset[pt_idx];

        // base
        char* base = d_base + pt_idx * MAX_GUESS_LEN;
        int base_len = d_base_len[pt_idx];

        // last segment
        int offset = d_last_seg_offset[pt_idx] + local_guess_idx * MAX_GUESS_LEN;
        char* last = d_last_seg + offset;
        int last_len = d_last_seg_len[d_last_seg_offset[pt_idx]/MAX_GUESS_LEN + local_guess_idx];

        // 输出
        char* out = d_out + idx * MAX_GUESS_LEN;
        int i = 0;
        for (; i < base_len; ++i) out[i] = base[i];
        for (int j = 0; j < last_len; ++j) out[i + j] = last[j];
        out[i + last_len] = '\0';
    }
}

// ================== 主体实现 ==================

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
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
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

// 动态任务量聚合优化版
void PriorityQueue::PopNext(int batch_pt_num)
{
    if (priority.empty()) return;
    // batch_pt_num 现在代表最小任务量阈值
    vector<PT> batch_pts;
    int total_guess_num = 0;
    int pt_count = 0;

    // 动态聚合PT直到任务量足够
    for (auto it = priority.begin(); it != priority.end(); ++it) {
        const PT& pt = *it;
        int guess_num = 1;
        if (!pt.max_indices.empty())
            guess_num = pt.max_indices.back(); // 最后一段的value数
        batch_pts.push_back(pt);
        total_guess_num += guess_num;
        pt_count++;
        if (total_guess_num >= batch_pt_num)
            break;
    }
    if (batch_pts.empty()) return;

    // GPU批量生成
    GenerateBatchGPU(batch_pts);

    // 生成新PT
    vector<PT> new_pts;
    for (PT& pt : batch_pts)
    {
        vector<PT> pts = pt.NewPTs();
        for (PT& npt : pts)
        {
            CalProb(npt);
            new_pts.emplace_back(std::move(npt));
        }
    }
    priority.erase(priority.begin(), priority.begin() + pt_count);

    // 插入新PT
    for (PT& pt : new_pts)
    {
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
}

// ================== GPU批量生成实现（全局一维线程优化，异步版） ==================
void PriorityQueue::GenerateBatchGPU(const vector<PT>& batch_pts)
{
    int batch_size = batch_pts.size();
    vector<string> base_vec(batch_size);
    vector<vector<string>> last_seg_vec(batch_size);
    vector<int> base_len(batch_size);
    vector<int> last_seg_count(batch_size);

    // 统计所有PT的base和最后一段
    for (int i = 0; i < batch_size; ++i) {
        const PT& pt = batch_pts[i];
        string base;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (seg_idx == pt.content.size() - 1) break;
            if (pt.content[seg_idx].type == 1)
                base += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                base += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 3)
                base += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx++;
        }
        base_vec[i] = base;
        base_len[i] = base.length();

        // 最后一段
        segment* a = nullptr;
        if (pt.content.back().type == 1)
            a = &m.letters[m.FindLetter(pt.content.back())];
        else if (pt.content.back().type == 2)
            a = &m.digits[m.FindDigit(pt.content.back())];
        else if (pt.content.back().type == 3)
            a = &m.symbols[m.FindSymbol(pt.content.back())];
        last_seg_vec[i] = a->ordered_values;
        last_seg_count[i] = a->ordered_values.size();
    }

    // 统计每个PT的猜测全局起始索引
    vector<int> pt_guess_offset(batch_size + 1, 0);
    for (int i = 0; i < batch_size; ++i)
        pt_guess_offset[i + 1] = pt_guess_offset[i] + last_seg_count[i];
    int total_guess_num = pt_guess_offset[batch_size];

    // 打包base
    vector<char> h_base(batch_size * MAX_GUESS_LEN, 0);
    for (int i = 0; i < batch_size; ++i)
        strncpy(&h_base[i * MAX_GUESS_LEN], base_vec[i].c_str(), MAX_GUESS_LEN - 1);

    // 打包last segment
    vector<char> h_last_seg;
    vector<int> h_last_seg_len;
    vector<int> h_last_seg_offset(batch_size);
    int last_offset = 0;
    for (int i = 0; i < batch_size; ++i) {
        h_last_seg_offset[i] = last_offset;
        for (const string& s : last_seg_vec[i]) {
            char buf[MAX_GUESS_LEN] = {0};
            strncpy(buf, s.c_str(), MAX_GUESS_LEN - 1);
            h_last_seg.insert(h_last_seg.end(), buf, buf + MAX_GUESS_LEN);
            h_last_seg_len.push_back(strlen(buf));
            last_offset += MAX_GUESS_LEN;
        }
    }

    // 2. 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 3. 拷贝到GPU（异步）
    char* d_base; int* d_base_len;
    char* d_last_seg; int* d_last_seg_len; int* d_last_seg_offset; int* d_last_seg_count;
    int* d_pt_guess_offset;
    char* d_out;
    cudaMalloc(&d_base, h_base.size());
    cudaMalloc(&d_base_len, batch_size * sizeof(int));
    cudaMalloc(&d_last_seg, h_last_seg.size());
    cudaMalloc(&d_last_seg_len, h_last_seg_len.size() * sizeof(int));
    cudaMalloc(&d_last_seg_offset, batch_size * sizeof(int));
    cudaMalloc(&d_last_seg_count, batch_size * sizeof(int));
    cudaMalloc(&d_pt_guess_offset, (batch_size + 1) * sizeof(int));
    cudaMalloc(&d_out, total_guess_num * MAX_GUESS_LEN);

    cudaMemcpyAsync(d_base, h_base.data(), h_base.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_base_len, base_len.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_last_seg, h_last_seg.data(), h_last_seg.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_last_seg_len, h_last_seg_len.data(), h_last_seg_len.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_last_seg_offset, h_last_seg_offset.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_last_seg_count, last_seg_count.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_pt_guess_offset, pt_guess_offset.data(), (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);

    // 4. 启动kernel（异步）
    int threads = 256;
    int blocks = (total_guess_num + threads - 1) / threads;
    generate_guesses_kernel_flat<<<blocks, threads, 0, stream>>>(
        d_base, d_base_len, d_last_seg, d_last_seg_len, d_last_seg_offset, d_last_seg_count,
        d_pt_guess_offset, total_guess_num, d_out
    );

    // 5. 异步回传结果
    vector<char> h_out(total_guess_num * MAX_GUESS_LEN, 0);
    cudaMemcpyAsync(h_out.data(), d_out, h_out.size(), cudaMemcpyDeviceToHost, stream);

    // 6. CPU可以在这里做其它工作（如准备下一批数据等）

    // 7. 等待流完成（需要用结果时才同步）
    cudaStreamSynchronize(stream);

    // 8. 存入guesses
    for (int i = 0; i < total_guess_num; ++i) {
        guesses.emplace_back(&h_out[i * MAX_GUESS_LEN]);
        total_guesses += 1;
    }

    // 9. 释放
    cudaFree(d_base); cudaFree(d_base_len);
    cudaFree(d_last_seg); cudaFree(d_last_seg_len);
    cudaFree(d_last_seg_offset); cudaFree(d_last_seg_count);
    cudaFree(d_pt_guess_offset);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
}

vector<PT> PT::NewPTs()
{
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
    CalProb(pt);
    if (pt.content.size() == 1)
    {
        segment *a;
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
        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
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
        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
    }
}