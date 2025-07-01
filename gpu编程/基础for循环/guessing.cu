#include "PCFG.h"
using namespace std;

#define BLOCK_SIZE 256
#define MAX_SEGMENT_VALUES 1000000
#define MAX_OUTPUT_GUESSES 1000000

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// CUDA内核：单segment
__global__ void generate_single_segment_kernel(
    GPUString* segment_values,
    int num_values,
    GPUString* output_guesses,
    int* output_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_values) {
        output_guesses[idx] = segment_values[idx];
        atomicAdd(output_count, 1);
    }
}

// CUDA内核：多segment
__global__ void generate_multi_segment_kernel(
    const char* base_guess,
    GPUString* segment_values,
    int num_values,
    GPUString* output_guesses,
    int* output_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_values) {
        GPUString result;
        result.set(base_guess);
        result.append(segment_values[idx].c_str());
        output_guesses[idx] = result;
        atomicAdd(output_count, 1);
    }
}

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

void PriorityQueue::PopNext()
{
    Generate(priority.front());
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

// ====== GPU优化的Generate实现（异步流式处理版）======
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    // 静态内存池，避免频繁分配和释放
    static GPUString* d_segment_values = nullptr;
    static GPUString* d_output_guesses = nullptr;
    static int* d_output_count = nullptr;
    static char* d_base_guess = nullptr;
    static bool gpu_initialized = false;
    static cudaStream_t stream;

    // Host端内存池
    static GPUString* h_segment_values_pool = nullptr;
    static GPUString* h_output_guesses_pool = nullptr;
    static size_t h_segment_values_pool_size = 0;
    static size_t h_output_guesses_pool_size = 0;

    if (!gpu_initialized) {
        CUDA_CHECK(cudaMalloc(&d_segment_values, MAX_SEGMENT_VALUES * sizeof(GPUString)));
        CUDA_CHECK(cudaMalloc(&d_output_guesses, MAX_OUTPUT_GUESSES * sizeof(GPUString)));
        CUDA_CHECK(cudaMalloc(&d_output_count, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_base_guess, MAX_STRING_LENGTH));
        // Host端内存池分配
        h_segment_values_pool = new GPUString[MAX_SEGMENT_VALUES];
        h_output_guesses_pool = new GPUString[MAX_OUTPUT_GUESSES];
        h_segment_values_pool_size = MAX_SEGMENT_VALUES;
        h_output_guesses_pool_size = MAX_OUTPUT_GUESSES;
        CUDA_CHECK(cudaStreamCreate(&stream));
        gpu_initialized = true;
    }

    if (pt.content.size() == 1)
    {
        segment *a = nullptr;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        if (pt.content[0].type == 3)
            a = &m.symbols[m.FindSymbol(pt.content[0])];

        int num_values = pt.max_indices[0];
        if (num_values > MAX_SEGMENT_VALUES) num_values = MAX_SEGMENT_VALUES;

        // 使用内存池
        GPUString* h_segment_values = h_segment_values_pool;
        for (int i = 0; i < num_values; i++)
            h_segment_values[i].set(a->ordered_values[i].c_str());

        CUDA_CHECK(cudaMemcpyAsync(d_segment_values, h_segment_values, num_values * sizeof(GPUString), cudaMemcpyHostToDevice, stream));
        int zero = 0;
        CUDA_CHECK(cudaMemcpyAsync(d_output_count, &zero, sizeof(int), cudaMemcpyHostToDevice, stream));

        int grid_size = (num_values + BLOCK_SIZE - 1) / BLOCK_SIZE;
        generate_single_segment_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            d_segment_values, num_values, d_output_guesses, d_output_count
        );

        GPUString* h_output_guesses = h_output_guesses_pool;
        CUDA_CHECK(cudaMemcpyAsync(h_output_guesses, d_output_guesses, num_values * sizeof(GPUString), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int i = 0; i < num_values; i++) {
            guesses.emplace_back(std::string(h_output_guesses[i].c_str()));
            total_guesses += 1;
        }
        // 不再delete
    }
    else
    {
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 2)
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 3)
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
                break;
        }

        segment *a = nullptr;
        if (pt.content[pt.content.size() - 1].type == 1)
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        if (pt.content[pt.content.size() - 1].type == 2)
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        if (pt.content[pt.content.size() - 1].type == 3)
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];

        int num_values = pt.max_indices[pt.content.size() - 1];
        
        if (num_values > MAX_SEGMENT_VALUES) num_values = MAX_SEGMENT_VALUES;
        GPUString* h_segment_values = h_segment_values_pool;
        for (int i = 0; i < num_values; i++)
            h_segment_values[i].set(a->ordered_values[i].c_str());

        CUDA_CHECK(cudaMemcpyAsync(d_segment_values, h_segment_values, num_values * sizeof(GPUString), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_base_guess, guess.c_str(), guess.length() + 1, cudaMemcpyHostToDevice, stream));
        int zero = 0;
        CUDA_CHECK(cudaMemcpyAsync(d_output_count, &zero, sizeof(int), cudaMemcpyHostToDevice, stream));

        int grid_size = (num_values + BLOCK_SIZE - 1) / BLOCK_SIZE;
        generate_multi_segment_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            d_base_guess, d_segment_values, num_values, d_output_guesses, d_output_count
        );

        GPUString* h_output_guesses = h_output_guesses_pool;
        CUDA_CHECK(cudaMemcpyAsync(h_output_guesses, d_output_guesses, num_values * sizeof(GPUString), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int i = 0; i < num_values; i++) {
            guesses.emplace_back(std::string(h_output_guesses[i].c_str()));
            total_guesses += 1;
        }
    }
}