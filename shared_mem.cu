#include "cuda_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "bit_functions.cuh"

#define EMPTY_WORD	0
#define FULL_WORD	1
#define TAIL_WORD	2

#define WARPS_IN_BLOCK 1

//__global__ void scan(float *g_odata, float *g_idata, int n)
//{
//	extern __shared__ float temp[]; // allocated on invocation
//	int thid = threadIdx.x;
//	int pout = 0, pin = 1;   
//	
//	// Load input into shared memory.
//	// This is exclusive scan, so shift right by one
//	// and set first element to 0
//	temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;
//	__syncthreads();
//	for (int offset = 1; offset < n; offset *= 2)
//	{     
//		pout = 1 - pout; // swap double buffer indices     
//		pin = 1 - pout;
//		if (thid >= offset)
//			temp[pout*n+thid] += temp[pin*n+thid - offset];
//		else
//			temp[pout*n+thid] = temp[pin*n+thid];
//		__syncthreads();
//	}   
//	g_odata[thid] = temp[pout*n+thid]; // write output
//} 

typedef struct segment {
	uchar1 l_end_type;
	uchar1 l_end_len;

	uchar1 r_end_type;
	uchar1 r_end_len;
} segment;

__global__ void SharedMemKernel(UINT* input, int inputSize, UINT* output)
{
	const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	const int lane_id = threadIdx.x % warpSize;
	const int warp_id = threadIdx.x / warpSize;
	const int warps_count = (inputSize % warpSize == 0) ? (inputSize / warpSize) : (inputSize / warpSize) + 1;

	UINT gulp = input[thread_id];

	// calculate type of the word
	bool is_zero = is_zeros(gulp);
	bool is_one = is_ones(gulp);
	char w_type;
	if (is_zero)
		w_type = EMPTY_WORD;
	else if (is_one)
		w_type = FULL_WORD;
	else
		w_type = TAIL_WORD;

	// is this thread the beginning of a section?
	char prev_type = __shfl_up_sync(FULL_MASK, w_type, 1);
	bool is_begin = false;
	if (thread_id < inputSize)
	{
		is_begin = (w_type == TAIL_WORD) || (w_type != prev_type);
		if (lane_id == 0)
			is_begin = true;
	}

	unsigned warp_begins_mask = __ballot_sync(FULL_MASK, is_begin);

	__shared__ segment segments[WARPS_IN_BLOCK];	// TODO: make it allocated dynamically
	int segment_len;
	if (is_begin)
	{																									// find ID of the next thread-beginning and thus the length of the section
		segment_len = (lane_id == warpSize - 1) ? 0 : __ffs(warp_begins_mask >> (lane_id + 1));			// note: bit shift "(int) >> 32" is not defined
																										// note: __ffs(0) = 0
		if (segment_len == 0)	// the last thread-beginning in warp
		{
			segment_len = (warp_id < warps_count - 1) ? (warpSize - lane_id) : (inputSize - thread_id);	// considers case of the last thread-beginning in the last warp in block
																										// when inputSize is not divisible by 32
			segments[warp_id].r_end_type = make_uchar1(w_type);
			segments[warp_id].r_end_len = make_uchar1(segment_len);
		}
		if (lane_id == 0)		// the first thread-beginning in warp
		{
			segments[warp_id].l_end_type = make_uchar1(w_type);
			segments[warp_id].l_end_len = make_uchar1(segment_len);
		}
	}
	__syncthreads();

	if (is_begin)
	{
		if (warp_id > 0 && lane_id == 0 && (segments[warp_id - 1].r_end_type.x == w_type))				// check if the first thread-beginning in warp is really
			is_begin = false;																			// thread-beginning in the context of the block...

		if (segment_len == 0)																			// ...if not, the last thread-beginning form prev. warp should add sth to its `segment_len`
		{
			for (int i = warp_id + 1; i < warps_count && segments[i].l_end_type.x == w_type; i++)
			{
				segment_len += segments[i].l_end_len.x;		// check types
				if (segments[i].l_end_len.x != warpSize)
					break;
			}
		}
	}
	// here every thread-beginning knows its segment's length (in-block boundaries)

	// in-warp scan, taken from https://github.com/NVIDIA/cuda-samples/blob/master/Samples/shfl_scan/shfl_scan.cu
	// not work-efficient implementation
	// TODO: do better implementation
	// TODO: scan should be exclusive
	int value = is_begin ? 1 : 0;
	for (int i = 1; i <= warpSize; i *= 2)
	{
		int n = __shfl_up_sync(FULL_MASK, value, i);

		if (lane_id >= i)
			value += n;
	}

	// inter-warp scan
	__shared__ int sums[WARPS_IN_BLOCK];
	if (lane_id == warpSize - 1)
		sums[warp_id] = value;
	__syncthreads();

	// the same shfl scan operation, but performed on warp sums
	// this can be safely done by a single warp
	if (warp_id == 0 && lane_id < warps_count)
	{
		int warp_sum = sums[lane_id];

		int mask = (1 << warps_count) - 1;
		for (int i = 1; i <= warps_count; i *= 2)
		{
			int n = __shfl_up_sync(mask, warp_sum, i, warps_count);
			if (lane_id >= i)
				warp_sum += n;
		}
		sums[lane_id] = warp_sum;
	}
	__syncthreads();



	if (is_begin)
	{
		// gather
		if (warp_id > 0)
			value += sums[warp_id - 1];

		if (w_type == EMPTY_WORD)
			output[value - 1] = get_compressed(segment_len, 0);
		else if (w_type == FULL_WORD)
			output[value - 1] = get_compressed(segment_len, 1);
		else
			output[value - 1] = gulp;
	}
}

void printBits(size_t const size, void const * const ptr)
{
	unsigned char *b = (unsigned char*)ptr;
	unsigned char byte;
	size_t i, j;

	for (i = size - 1; i >= 0; i--) {
		for (j = 7; j >= 0; j--) {
			byte = (b[i] >> j) & 1;
			printf("%u", byte);
		}
	}
	puts("");
}

UINT* SharedMemWAH(int size, UINT* input)//, size_t size)
{
	UINT* result = nullptr;

	UINT* d_input;
	UINT* d_output;
	CUDA_CHECK(cudaMalloc((void**)&d_input, size * sizeof(UINT)), FreeInput);		// reinterpret_cast<>
	CUDA_CHECK(cudaMalloc((void**)&d_output, size * sizeof(UINT)), Free);
	CUDA_CHECK(cudaMemcpy(d_input, input, size * sizeof(UINT), cudaMemcpyHostToDevice), Free);

	int threads_per_block = 64;
	int blocks = size / threads_per_block;
	if (size % threads_per_block != 0)
		blocks++;

	SharedMemKernel<<<blocks, threads_per_block>>>(d_input, size, d_output);

	CUDA_CHECK(cudaGetLastError(), Free);
	CUDA_CHECK(cudaDeviceSynchronize(), Free);

	UINT* output = new UINT[size];
	CUDA_CHECK(cudaMemcpy(output, d_output, size * sizeof(UINT), cudaMemcpyDeviceToHost), Free);
	result = output;

Free:
	CUDA_CHECK(cudaFree(d_output), FreeInput);
FreeInput:
	CUDA_CHECK(cudaFree(d_input), Fin);
Fin:
	return result;
}
