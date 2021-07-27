#include <cstdio>
#include <climits>
#include <thrust/remove.h>
#include "bit_functions.cuh"
#include "defines.h"
#include "methods.h"

//maximum input for logging
#ifndef LOGGING_MAX
#define LOGGING_MAX 0
#endif

__global__ void ballot_warp_compress(UINT* input, UINT* output)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_id % 32;

    bool is_zero = is_zeros(input[global_id]);
    bool is_one = is_ones(input[global_id]);

    UINT zeros = __ballot_sync(0xffffffff, is_zero);
    UINT ones = __ballot_sync(0xffffffff, is_one);
    zeros = reverse(zeros);
    ones = reverse(ones);
    bool checks_next = true;
    if (!is_zero && !is_one) //not suitable for compression
    {
        checks_next = false;
    }
    else
    {
        if (warp_id > 0)
        {
            int previous_zero = get_bit(zeros, warp_id-1);
            int previous_one = get_bit(ones, warp_id-1);
            if (previous_zero && is_zero) checks_next = false;
            else if (previous_one && is_one) checks_next = false;
        }
    }
    UINT add = 0;
    if (checks_next)
    {
        int pos = warp_id;
        while (pos <= 31)
        {
            int next_zero = get_bit(zeros, pos);
            int next_one = get_bit(ones, pos);
            if (is_zero && next_zero) add++;
            else if (is_one && next_one) add++;
            else break;
            pos++;
        }
        output[global_id] = get_compressed(add, is_one);
    }
    else
    {
        if (!is_zero && !is_one)output[global_id] = input[global_id]; //cant compress, writing literally
        else output[global_id] = 0; //'null' symbol after compression
    }
}

__global__ void warm_up() 
{

}

__global__ void single_ballot_warp_compress(UINT* input, UINT* output)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_id % 32;

    bool is_zero = is_zeros(input[global_id]);
    bool is_one = is_ones(input[global_id]);

    UINT zeros = __ballot_sync(0xffffffff, is_zero);
    UINT ones = __ballot_sync(0xffffffff, is_one);
    output[global_id] = 0;
    zeros = reverse(zeros);
    ones = reverse(ones);
    
    if (warp_id == 0)
    {
        int which = -1;
        int sum = 0;
        int write_pos = 0;
        for (int i = 0; i < 32; i++)
        {
            int zero = get_bit(zeros, i);
            int one = get_bit(ones, i);
            if (!zero && !one)
            {
                if (sum > 0)
                {
                    output[global_id + write_pos] = get_compressed(sum, which);
                    write_pos++;
                    sum = 0;
                    which = -1;
                }
                output[global_id + write_pos] = input[global_id + i];
                write_pos++;
            }
            else if (zero)
            {
                if (which == -1)
                {
                    sum = 1;
                    which = 0;
                }
                else if (which == 1)
                {
                    output[global_id + write_pos] = get_compressed(sum, which);
                    write_pos++;
                    sum = 1;
                    which = 0;
                }
                else sum++;
            }
            else if (one)
            {
                if (which == -1)
                {
                    sum = 1;
                    which = 1;
                }
                else if (which == 0)
                {
                    output[global_id + write_pos] = get_compressed(sum, which);
                    write_pos++;
                    sum = 1;
                    which = 1;
                }
                else sum++;
            }
        }
        if (sum > 0)
        {
            output[global_id + write_pos] = get_compressed(sum, which);
        }
    }
}

__global__ void ballot_warp_merge(int input_size, UINT* input, UINT* output)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    UINT curr = input[global_id];
    output[global_id] = 0;
    bool checks_next = true;
    if (!get_bit(curr, 0)) checks_next = false;
    else if (global_id > 0)
    {
        UINT prev = input[global_id - 1];
        if (get_bit(prev, 0) && (get_bit(prev, 1) == get_bit(curr, 1)))
        {
            checks_next = false;
        }
        if (curr == 0)
        {
            checks_next = false;
        }
    }

    if (checks_next)
    {
        UINT bit = get_bit(curr, 1);
        int curr_output_pos = global_id;    // ???
        int pos = global_id + 1;
        UINT currAmount = compressed_count(curr);
        while (pos < input_size)
        {
            UINT iter = input[pos];
            if (get_bit(iter, 0) == 0) break;
            if (get_bit(iter, 1) != bit) break;
            UINT added = compressed_count(iter);
            currAmount += added; 
            pos++;
        }
        if (currAmount > 0)
        {
            output[curr_output_pos] = get_compressed(currAmount, bit);
        }
    }
    else if (!get_bit(curr, 0))
    {
        output[global_id] = curr;
    }
}

__global__ void atomic_sum_warp_merge(int input_size, UINT* input, UINT* output)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= input_size) return;
    UINT curr = input[global_id];
    UINT bit = get_bit(curr, 1);
    bool is_compressed = get_bit(curr, 0);
    if (!is_compressed) output[global_id] == curr;
    else output[global_id] = compressed_count(curr);

    if (is_compressed && global_id>0)
    {
        bool is_first = false;
        int compress_pos = global_id;
        int pos = global_id;
        while (compress_pos > 0)
        {
            pos--;
            if (get_bit(input[pos], 0) && get_bit(input[pos], 1) == bit)
            {
                compress_pos = pos;
            }
            else if (input[pos] != 0)
            {
                if (global_id == compress_pos) is_first = true;
                break;
            }
        }
        if (!is_first)
        {
            output[global_id] = 0;
            int add_pos = compress_pos;
            UINT to_add = compressed_count(curr);
            while (to_add > 0)
            {
                UINT curr_sum = output[add_pos];
                if ((unsigned long long)(to_add + curr_sum) > COMPRESS_MAX)
                {
                    UINT to_fill = COMPRESS_MAX - curr_sum;
                    atomicAdd(&output[add_pos], to_fill);
                    to_add -= to_fill;
                    add_pos++;
                }
                else
                {
                    atomicAdd(&output[add_pos], to_add);
                    to_add = 0;
                }
            }
        }
    }
}

__global__ void atomic_sum_warp_write(int input_size, UINT* input, UINT* output)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= input_size) return;
    UINT curr = input[global_id];
    UINT bit = get_bit(curr, 1);
    bool is_compressed = get_bit(curr, 0);
    if (is_compressed)
    {
        int pos = global_id;
        while (pos < input_size)
        {
            if (output[pos] > 0)
            {
                if (pos == global_id || input[pos] == 0)
                {
                    output[pos] = get_compressed(output[pos], bit);
                }
                else break;
            }
            else break;
            pos++;
        }
    }
}


UINT* BallotSyncWAH(int data_size, UINT * d_input)
{
    int size = data_size;
    UINT* output = new UINT[size];
    UINT* d_output;

    cudaMalloc((UINT**)&d_output, sizeof(UINT) * size);
    ballot_warp_compress<<<(size / GPU_THREADS_COUNT)+1, GPU_THREADS_COUNT>>>(d_input,d_output);

    cudaMemcpy(output, d_output, sizeof(UINT) * size, cudaMemcpyDeviceToHost);

    UINT* end = thrust::remove_if(output, output + size, wah_zero());
    int remove_count = end - output;

    if (size <= LOGGING_MAX)
    {
        printf("Sequence after warp-compression:\n");
        for (int i = 0; i < end - output; i++)
        {
            UINT c = compressed_count(output[i]);
            if (get_bit(output[i], 0)) printf("(%u,%u) ", c, get_bit(output[i], 1));
            else printf("x ");
        }
        printf("\n");
    }

    cudaMemcpy(d_input, output, sizeof(UINT) * remove_count, cudaMemcpyHostToDevice);

    ballot_warp_merge <<<(remove_count / GPU_THREADS_COUNT)+1, GPU_THREADS_COUNT >> > (remove_count, d_input, d_output);
    cudaMemcpy(output, d_output, sizeof(UINT) * size, cudaMemcpyDeviceToHost);
    UINT* final_end = thrust::remove_if(output, output + remove_count, wah_zero());
    if (size <= LOGGING_MAX)
    {
        printf("Sequence after global-compression:\n");
        for (int i = 0; i < final_end - output; i++)
        {
            UINT c = compressed_count(output[i]);
            if (get_bit(output[i], 0)) printf("(%u,%u) ", c, get_bit(output[i], 1));
            else printf("x ");
        }
        printf("\n");
    }
    cudaFree(d_input);
    cudaFree(d_output);
    return output;
}

UINT* AtomicAddWAH(int data_size, UINT* d_input)
{
    int size = data_size;
    UINT* output = new UINT[size];
    UINT* d_output;

    cudaMalloc((UINT**)&d_output, sizeof(UINT) * size);
    single_ballot_warp_compress << <size / 32, 32 >> > (d_input, d_output);

    cudaMemcpy(output, d_output, sizeof(UINT) * size, cudaMemcpyDeviceToHost);

    if (size <= LOGGING_MAX)
    {
        printf("Sequence after warp-compression:\n");
        for (int i = 0; i < size; i++)
        {
            UINT c = compressed_count(output[i]);
            if (get_bit(output[i], 0)) printf("(%u,%u) ", c, get_bit(output[i], 1));
            else printf("x ");
        }
        printf("\n");
    }

    cudaMemcpy(d_input, output, sizeof(UINT) * size, cudaMemcpyHostToDevice);

    atomic_sum_warp_merge << <(size / 32) + 1, 32 >> > (size, d_input, d_output);
    atomic_sum_warp_write << <(size / 32) + 1, 32 >> > (size, d_input, d_output);
    cudaMemcpy(output, d_output, sizeof(UINT) * size, cudaMemcpyDeviceToHost);
    UINT* final_end = thrust::remove_if(output, output + size, wah_zero());
    if (size <= LOGGING_MAX)
    {
        printf("Sequence after global-compression:\n");
        for (int i = 0; i < final_end - output; i++)
        {
            UINT c = compressed_count(output[i]);
            if (get_bit(output[i], 0)) printf("(%u,%u) ", c, get_bit(output[i], 1));
            else printf("x ");
        }
        printf("\n");
    }
    cudaFree(d_input);
    cudaFree(d_output);
    return output;
}

void WarmUp()
{
    warm_up << <1, 1 >> > ();
}