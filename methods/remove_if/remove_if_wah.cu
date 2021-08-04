#include <cstdio>
#include <climits>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include "../../bit_functions.cuh"
#include "../../defines.h"
#include "../../methods.h"

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

__global__ void optimized_ballot_warp_compress(UINT* input, UINT* output)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_id % 32;

    bool is_zero = is_zeros(input[global_id]);
    bool is_one = is_ones(input[global_id]);

    UINT zeros = __ballot_sync(0xffffffff, is_zero);
    UINT ones = __ballot_sync(0xffffffff, is_one);
    bool checks_next = true;
    if (!is_zero && !is_one) //not suitable for compression
    {
        checks_next = false;
    }
    else
    {
        if (warp_id > 0)
        {
            int previous_zero = get_reversed_bit(zeros, warp_id - 1);
            int previous_one = get_reversed_bit(ones, warp_id - 1);
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
            int next_zero = get_reversed_bit(zeros, pos);
            int next_one = get_reversed_bit(ones, pos);
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


UINT* RemoveIfWAH(int data_size, UINT * d_input, int threads_per_block)
{
    int size = data_size;
    UINT* output = new UINT[size];
    UINT* d_output;

    cudaMalloc((UINT**)&d_output, sizeof(UINT) * size);
    ballot_warp_compress<<<(size / threads_per_block)+1, threads_per_block >>>(d_input,d_output);

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

    ballot_warp_merge <<<(remove_count / threads_per_block)+1, threads_per_block >> > (remove_count, d_input, d_output);
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

UINT* OptimizedRemoveIfWAH(int data_size, UINT* d_input, int threads_per_block)
{
    int size = data_size;
    UINT* output = new UINT[size];
    UINT* d_output;

    cudaMalloc((UINT**)&d_output, sizeof(UINT) * size);
    optimized_ballot_warp_compress << <(size / threads_per_block) + 1, threads_per_block >> > (d_input, d_output);

    UINT* end = thrust::remove_if(thrust::device, d_output, d_output + size, wah_zero());

    int remove_count = end - d_output;

    ballot_warp_merge << <(remove_count / threads_per_block) + 1, threads_per_block >> > (remove_count, d_output, d_input);

    UINT* final_end = thrust::remove_if(thrust::device, d_input, d_input + remove_count, wah_zero());
    int final_count = final_end - d_input;

    cudaMemcpy(output, d_input, sizeof(UINT) * final_count, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    return output;
}

void WarmUp()
{
    warm_up << <1, 1 >> > ();
}