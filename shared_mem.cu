#include "cuda_runtime.h"
#include "cooperative_groups.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "bit_functions.cuh"

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


// taken from https://github.com/NVIDIA/cuda-samples/blob/master/Samples/shfl_scan/shfl_scan.cu
// not work-efficient implementation
// TODO: do better implementation
// TODO: scan should be exclusive
__device__ void inclusive_scan_inblock_sync(int* value, int* smem_ptr, int warp_id, int lane_id, int warps_count)
{
	// in-warp scan
	for (int i = 1; i <= warpSize; i *= 2)
	{
		int n = __shfl_up_sync(FULL_MASK, *value, i);	// add width as argument???

		if (lane_id >= i)
			*value += n;
	}
	if (warps_count == 1)
		return;

	// inter-warp scan
	if (lane_id == warpSize - 1)
		smem_ptr[warp_id] = *value;
	__syncthreads();

	// the same shfl scan operation, but performed on warp sums
	// this can be safely done by a single warp, since there is maximum of 32 warps in a block
	if (warp_id == 0 && lane_id < warps_count)
	{
		int warp_sum = smem_ptr[lane_id];

		int mask = (1 << warps_count) - 1;
		for (int i = 1; i <= warps_count; i *= 2)
		{
			int n = __shfl_up_sync(mask, warp_sum, i);
			if (lane_id >= i)
				warp_sum += n;
		}
		smem_ptr[lane_id] = warp_sum;
	}
	__syncthreads();

	if (warp_id > 0)
		*value += smem_ptr[warp_id - 1];
	__syncthreads();
}

// kernel assumes that grid is 1D
__global__ void SharedMemKernel(UINT* input, int inputSize, UINT* output)
{
	const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	const int lane_id = threadIdx.x % warpSize;
	const int warp_id = threadIdx.x / warpSize;
	int warps_count;// = (inputSize % warpSize == 0) ? (inputSize / warpSize) : (inputSize / warpSize) + 1;	// TODO: correct for when there are many blocks
	if ((blockIdx.x + 1)*blockDim.x > inputSize)	// last block can enter here
	{
		warps_count = (inputSize - blockIdx.x*blockDim.x) / warpSize;
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
	int segment_len;
	if (is_begin)
	{																									// find ID of the next thread-beginning and thus the length of the section
		segment_len = (lane_id == warpSize - 1) ? 0 : __ffs(warp_begins_mask >> (lane_id + 1));			// note: bit shift "(int) >> 32" is not defined
																										// note: __ffs(0) = 0
		if (segment_len == 0)	// the last thread-beginning in warp
		{
			am_last_beginning_inwarp = true;
			segment_len = (warp_id < warps_count - 1) ? (warpSize - lane_id) : (warps_count*warpSize - threadIdx.x);
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


	__shared__ int sums[WARPS_IN_BLOCK];
	int index = is_begin ? 1 : 0;
	inclusive_scan_inblock_sync(&index, sums, warp_id, lane_id, warps_count);
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

	__shared__ bool decrement_index;
	if (threadIdx.x == 0)
		decrement_index = false;
	if (is_begin)
	{
		if (blockIdx.x > 0 && threadIdx.x == 0 && w_type != TAIL_WORD &&								// check if the first thread-beginning in block is really
			block_segments[blockIdx.x - 1].r_end_type.x == w_type)										// thread-beginning in the context of the grid...
		{
			is_begin = false;
			am_last_beginning_inblock = false;
			decrement_index = true;
		}

		if (am_last_beginning_inblock)																	// ...if not, the last thread-beginning form prev. block should add sth to its `segment_len`
		{
			for (int i = blockIdx.x + 1; i < gridDim.x && block_segments[i].l_end_type.x == w_type; i++)
			{
				segment_len += block_segments[i].l_end_len.x;		// check types
				if (block_segments[i].l_end_len.x != blockDim.x)
					break;
			}
		}
	}
	__syncthreads();
	if (decrement_index)
		index--;
	grid.sync();



	// INTER-BLOCKS SCAN
	// write block_sum to global memory
	UINT* block_sums = output;
	__shared__ bool has_last_beginning;
	if (threadIdx.x == 0)
		has_last_beginning = false;
	__syncthreads();
	if (am_last_beginning_inblock)
	{
		has_last_beginning = true;
		block_sums[blockIdx.x] = index;
	}
	__syncthreads();
	if (!has_last_beginning)
	{
		if (threadIdx.x == warps_count * warpSize - 1)
			block_sums[blockIdx.x] = index;
	}
	grid.sync();


	// Kernel assumes that there are at least as many threads in block as the total number of blocks.
	// This assumption makes sense since this kernel is cooperative.
	// Indeed, there ain't many blocks then.
	__shared__ int partial_sums[2048];	//TODO: change
	int block_sum = 0;
	if (thread_id < gridDim.x)
		block_sum = block_sums[thread_id];
	grid.sync();

	inclusive_scan_inblock_sync(&block_sum, partial_sums, warp_id, lane_id, warps_count);

	if (thread_id < gridDim.x)
		block_sums[thread_id] = block_sum;
	grid.sync();

	if (is_begin)
	{
		int ind = blockIdx.x;
		if (blockIdx.x > 0)
			index += block_sums[ind - 1];
	}

	if (is_begin)
	{
		if (w_type == EMPTY_WORD)
			output[index - 1] = get_compressed(segment_len, 0);
		else if (w_type == FULL_WORD)
			output[index - 1] = get_compressed(segment_len, 1);
		else
			output[index - 1] = gulp;
	}
}

void ensure_cooperativity_support()
{
	cudaDeviceProp deviceProp = { 0 };

	int device;
	CUDA_CHECK(cudaGetDevice(&device), Finish);

	CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device), Finish);
	if (!deviceProp.cooperativeLaunch)
	{
		printf("\nSelected GPU (%d) does not support Cooperative Kernel Launch, Waiving the run\n", device);
		exit(EXIT_FAILURE);
	}
Finish:
}

UINT* SharedMemWAH(int size, UINT* input)
{
	ensure_cooperativity_support();

	UINT* result = nullptr;

	UINT* d_input;
	UINT* d_output;
	CUDA_CHECK(cudaMalloc((void**)&d_input, size * sizeof(UINT)), FreeInput);		// reinterpret_cast<>
	CUDA_CHECK(cudaMalloc((void**)&d_output, size * sizeof(UINT)), Free);
	CUDA_CHECK(cudaMemcpy(d_input, input, size * sizeof(UINT), cudaMemcpyHostToDevice), Free);

	int threads_per_block = 32;
	int blocks = size / threads_per_block;
	if (size % threads_per_block != 0)
		blocks++;

	// as many blocks as there are SMs can be launched safely
	int device = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);
	//printf("SMs: %d\n", deviceProp.multiProcessorCount);

	int numBlocksPerSm = 0;
	//int numThreads = 128;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, SharedMemKernel, threads_per_block, 0);
	//printf("numBlocksPerSm: %d \n", numBlocksPerSm);

	//SharedMemKernel<<<blocks, threads_per_block>>>(d_input, size, d_output);
	void* params[3];
	params[0] = &d_input;
	params[1] = &size;
	params[2] = &d_output;
	cudaLaunchCooperativeKernel((void*)SharedMemKernel, blocks, threads_per_block, params);

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
