#include "cuda_runtime.h"
#include "cooperative_groups.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "../../bit_functions.cuh"

//#define EMPTY_WORD	0
//#define FULL_WORD	1
//#define TAIL_WORD	2

#define WARPS_IN_BLOCK 32

namespace cg = cooperative_groups;

typedef struct segment {
	uchar1 l_end_type;
	uchar1 l_end_len;

	uchar1 r_end_type;
	uchar1 r_end_len;
} segment;

__global__ void SharedMemKernelBlocks(UINT* input, int inputSize, UINT* output)
{
	const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	const int lane_id = threadIdx.x % warpSize;
	const int warp_id = threadIdx.x / warpSize;
	int warps_count;// = (inputSize % warpSize == 0) ? (inputSize / warpSize) : (inputSize / warpSize) + 1;	// TODO: correct for when there are many blocks
	if ((blockIdx.x + 1) * blockDim.x > inputSize)	// last block can enter here
	{
		warps_count = (inputSize - blockIdx.x * blockDim.x) / warpSize;
		if (inputSize % warpSize != 0)
			warps_count++;
	}
	else
		warps_count = blockDim.x / warpSize;

	UINT gulp = input[thread_id];

	WORD_TYPE w_type = get_word_type(gulp);

	// is this thread the beginning of a section?
	bool is_begin = false;
	char prev_type = __shfl_up_sync(FULL_MASK, w_type, 1);
	if (thread_id < inputSize)
	{
		is_begin = (w_type == TAIL_WORD) || (w_type != prev_type);
		if (lane_id == 0)
			is_begin = true;
	}

	unsigned warp_begins_mask = __ballot_sync(FULL_MASK, is_begin);
	bool am_last_beginning_inwarp = false;

	__shared__ segment segments[WARPS_IN_BLOCK];	// TODO: make it allocated dynamically
	__shared__ bool has_last_beginning_inblock;

	has_last_beginning_inblock = false;
	int segment_len;
	if (is_begin)
	{																									// find ID of the next thread-beginning and thus the length of the section
		segment_len = (lane_id == warpSize - 1) ? 0 : __ffs(warp_begins_mask >> (lane_id + 1));			// note: bit shift "(int) >> 32" is not defined
																										// note: __ffs(0) = 0
		if (segment_len == 0)	// the last thread-beginning in warp
		{
			am_last_beginning_inwarp = true;
			segment_len = (warp_id < warps_count - 1) ? (warpSize - lane_id) : (warps_count * warpSize - thread_id);
			// considers case of the last thread-beginning in the last warp in block
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
		if (warp_id > 0 && lane_id == 0 && w_type != TAIL_WORD &&										// check if the first thread-beginning in warp is really
			(segments[warp_id - 1].r_end_type.x == w_type))												// thread-beginning in the context of the block...
		{
			is_begin = false;
			am_last_beginning_inwarp = false;
		}

		if (am_last_beginning_inwarp)																	// ...if not, the last thread-beginning form prev. warp should add sth to its `segment_len`
		{
			for (int i = warp_id + 1; i < warps_count && segments[i].l_end_type.x == w_type; i++)
			{
				segment_len += segments[i].l_end_len.x;		// check types
				if (segments[i].l_end_len.x != warpSize)
					break;
			}
		}
	}
	__syncthreads();

	// here every thread-beginning knows its segment's length (in-block boundaries)

	// in-warp scan, taken from https://github.com/NVIDIA/cuda-samples/blob/master/Samples/shfl_scan/shfl_scan.cu
	// not work-efficient implementation
	// TODO: do better implementation
	// TODO: scan should be exclusive
	int index = is_begin ? 1 : 0;
	for (int i = 1; i <= warpSize; i *= 2)
	{
		int n = __shfl_up_sync(FULL_MASK, index, i);	// add width as argument???

		if (lane_id >= i)
			index += n;
	}

	// inter-warp scan
	__shared__ int sums[WARPS_IN_BLOCK];
	if (lane_id == warpSize - 1)
		sums[warp_id] = index;
	__syncthreads();

	// the same shfl scan operation, but performed on warp sums
	// this can be safely done by a single warp
	if (warp_id == 0 && lane_id < warps_count)
	{
		int warp_sum = sums[lane_id];
		//printf("thread %d has value %d\n", thread_id, warp_sum);
		//if (lane_id == 0)
		//	printf("------------------------\n");

		int mask = (1 << warps_count) - 1;
		for (int i = 1; i <= warps_count; i *= 2)
		{
			int n = __shfl_up_sync(mask, warp_sum, i);
			if (lane_id >= i)
				warp_sum += n;
		}
		sums[lane_id] = warp_sum;
	}
	__syncthreads();

	if (warp_id > 0 && is_begin)
		index += sums[warp_id - 1];
	// now index is correct in block boundaries


	// ================
	// INTER-BLOCKS STEP
	// ================
	segment* block_segments = (segment*)output;															// this allocation is just being reused
	// find the last thread-beginning in block
	warp_begins_mask = __ballot_sync(FULL_MASK, is_begin);
	__shared__ unsigned block_begins_masks[WARPS_IN_BLOCK];
	if (lane_id == 0)
		block_begins_masks[warp_id] = warp_begins_mask;
	__syncthreads();

	if (threadIdx.x < warps_count)																		// find last warp in block which contains any thread-beginning
	{
		unsigned begins_mask = block_begins_masks[threadIdx.x];
		unsigned is_mask_nonzero = __ballot_sync(__activemask(), begins_mask != EMPTY_MASK);
		if (threadIdx.x == 0)
			block_begins_masks[0] = warpSize - 1 - __clz(is_mask_nonzero);								// write its warp_id in shared memory
	}
	__syncthreads();

	bool am_last_beginning_inblock = false;
	if (warp_id == block_begins_masks[0])																// now we find last thread-beginning in block (in previously found warp)
	{
		int lane = warpSize - 1 - __clz(warp_begins_mask);	// lane_id of this thread
		if (lane_id == lane)
		{
			am_last_beginning_inblock = true;

			block_segments[blockIdx.x].r_end_type = make_uchar1(w_type);
			block_segments[blockIdx.x].r_end_len = make_uchar1(segment_len);
		}
	}
	if (threadIdx.x == 0)						// first thread-beginning in block
	{
		block_segments[blockIdx.x].l_end_type = make_uchar1(w_type);
		block_segments[blockIdx.x].l_end_len = make_uchar1(segment_len);
	}
	cg::grid_group grid = cg::this_grid();
	grid.sync();

	if (is_begin)
	{
		if (blockIdx.x > 0 && threadIdx.x == 0 && w_type != TAIL_WORD &&									// check if the first thread-beginning in block is really
			block_segments[blockIdx.x - 1].r_end_type.x == w_type)											// thread-beginning in the context of the grid...
		{
			is_begin = false;
			am_last_beginning_inblock = false;
		}

		if (am_last_beginning_inblock)																		// ...if not, the last thread-beginning form prev. block should add sth to its `segment_len`
		{
			for (int i = blockIdx.x + 1; i < gridDim.x && block_segments[i].l_end_type.x == w_type; i++)
			{
				segment_len += block_segments[i].l_end_len.x;		// check types
				if (segments[i].l_end_len.x != blockDim.x)
					break;
			}
		}
	}
	if (am_last_beginning_inblock) has_last_beginning_inblock = true;

	index += blockDim.x * blockIdx.x;
	if (is_begin)
	{
		// gather
		//if (warp_id > 0)
		//	index += sums[warp_id - 1];
		if (w_type == EMPTY_WORD)
			output[index - 1] = get_compressed(segment_len, 0);
		else if (w_type == FULL_WORD)
			output[index - 1] = get_compressed(segment_len, 1);
		else
			output[index - 1] = gulp;
	}
	else if (am_last_beginning_inblock)
	{
		while (index - 1 < blockDim.x)
		{
			output[index] = EMPTY_WORD;
			index++;
		}
	}
	if (!has_last_beginning_inblock)
	{
		output[index - 1] = EMPTY_WORD;
	}


}

UINT* RemoveIfSharedMemWAH(int size, UINT* input)
{

	UINT* result = nullptr;

	UINT* d_input;
	UINT* d_output;
	CUDA_CHECK(cudaMalloc((void**)&d_input, size * sizeof(UINT)), FreeInput);		// reinterpret_cast<>
	CUDA_CHECK(cudaMalloc((void**)&d_output, size * sizeof(UINT)), Free);
	CUDA_CHECK(cudaMemcpy(d_input, input, size * sizeof(UINT), cudaMemcpyHostToDevice), Free);

	int threads_per_block = 256;
	int blocks = size / threads_per_block;
	if (size % threads_per_block != 0)
		blocks++;

	//SharedMemKernel<<<blocks, threads_per_block>>>(d_input, size, d_output);
	void* params[3];
	params[0] = &d_input;
	params[1] = &size;
	params[2] = &d_output;
	cudaLaunchCooperativeKernel((void*)SharedMemKernelBlocks, blocks, threads_per_block, params);

	CUDA_CHECK(cudaGetLastError(), Free);
	CUDA_CHECK(cudaDeviceSynchronize(), Free);

	UINT* end = thrust::remove_if(thrust::device, d_output, d_output + size, wah_zero());

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