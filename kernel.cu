#include <cstdio>
#include <climits>
#include <thrust/remove.h>
#include "bit_functions.cuh"

struct zero
{
	__host__ __device__
		bool operator()(const int x)
	{
		return x == 0;
	}
};

__global__ void cuda_hello(){
    //printf("%d\n",blockIdx.x);
}


void CudaHello()
{
    printf("Hello extern!\n");
    cuda_hello<<<4,4>>>(); 
}


__global__ void ballot_warp_compress(UINT* input, UINT* output)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_id % 32;
    int warp_number = global_id / 32;

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


void BallotSyncWAH(UINT * input)
{
    int testSize = 64;
    UINT* test = new UINT[testSize];
    UINT* output = new UINT[testSize];
    for (int i = 0; i < testSize; i++)
    {
        int roll = rand() % 3;
        if (roll == 0)
        {
            test[i] = 0x7FFFFFFF; //all ones
            printf("1");
        }
        if (roll == 1)
        {
            test[i] = 256; // not valid for compression
            printf("x");
        }
        if (roll == 2)
        {
            test[i] = 0x00000000; // all zeros
            printf("0");
        }
    }
    printf("\n");
    UINT * d_test;
    UINT* d_output;
    cudaMalloc((UINT**)&d_test, sizeof(UINT) * testSize);
    cudaMalloc((UINT**)&d_output, sizeof(UINT) * testSize);
    cudaMemcpy(d_test, test, sizeof(UINT)*testSize, cudaMemcpyHostToDevice);
    ballot_warp_compress<<<testSize / 32, 32>>>(d_test,d_output);

    cudaMemcpy(output, d_output, sizeof(UINT) * testSize, cudaMemcpyDeviceToHost);

    UINT* end = thrust::remove_if(output, output + testSize, zero());

    for (int i = 0; i < end-output; i++)
    {
        printf("%u ", output[i]);
    }
    printf("\n");
    cudaFree(d_test);
    cudaFree(d_output);
    delete test;
    delete output;
}
