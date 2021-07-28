#include <cstdio>
#include <climits>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include "../../bit_functions.cuh"
#include "../../defines.h"
#include "../../methods.h"

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

__global__ void optimized_single_ballot_warp_compress(UINT* input, UINT* output)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_id % 32;

    bool is_zero = is_zeros(input[global_id]);
    bool is_one = is_ones(input[global_id]);

    UINT zeros = __ballot_sync(0xffffffff, is_zero);
    UINT ones = __ballot_sync(0xffffffff, is_one);
    output[global_id] = 0;
    if (warp_id == 0)
    {
        int which = -1;
        int sum = 0;
        int write_pos = 0;
        for (int i = 0; i < 32; i++)
        {
            int zero = get_reversed_bit(zeros, i);
            int one = get_reversed_bit(ones, i);
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

__global__ void atomic_sum_warp_merge(int input_size, UINT* input, UINT* output)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= input_size) return;
    UINT curr = input[global_id];
    UINT bit = get_bit(curr, 1);
    bool is_compressed = get_bit(curr, 0);
    if (!is_compressed) output[global_id] == curr;
    else output[global_id] = compressed_count(curr);

    if (is_compressed && global_id > 0)
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

UINT* AtomicAddWAH(int data_size, UINT* d_input, int threads_per_block)
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

UINT* OptimizedAtomicAddWAH(int data_size, UINT* d_input, int threads_per_block)
{
    int size = data_size;
    UINT* output = new UINT[size];
    UINT* d_output;

    cudaMalloc((UINT**)&d_output, sizeof(UINT) * size);
    optimized_single_ballot_warp_compress << <size / 32, 32 >> > (d_input, d_output);

    cudaMemcpy(d_input, d_output, sizeof(UINT) * size, cudaMemcpyDeviceToDevice);

    atomic_sum_warp_merge << <(size / 32) + 1, 32 >> > (size, d_input, d_output);
    atomic_sum_warp_write << <(size / 32) + 1, 32 >> > (size, d_input, d_output);

    UINT* final_end = thrust::remove_if(thrust::device, d_output, d_output + size, wah_zero());
    cudaMemcpy(output, d_output, sizeof(UINT) * size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    return output;
}
