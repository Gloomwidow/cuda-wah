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

enum WORD_TYPE {
	EMPTY_WORD = 0,
	FULL_WORD = 1,
	TAIL_WORD = 2
};


namespace cg = cooperative_groups;

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


// calculate type of the word
__device__ WORD_TYPE get_word_type(UINT gulp)
{
	if (is_zeros(gulp))
		return EMPTY_WORD;
	if (is_ones(gulp))
		return FULL_WORD;
	return TAIL_WORD;
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

	if (warp_id > 0)
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
				int tmp = segment_len;
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
	//__shared__ int sums[WARPS_IN_BLOCK];
	UINT* block_sums = output;
	//if (threadIdx.x == blockDim.x - 1)
	int writing_thread_id;
	__shared__ bool has_last_beginning;
	if (threadIdx.x == 0)
		has_last_beginning = false;
	__syncthreads();
	if (am_last_beginning_inblock)
	{
		has_last_beginning = true;
		//printf("block %d has value: %d\n", blockIdx.x, index);
		block_sums[blockIdx.x] = index;
		//printf("thread_id: %d, block_sum: %d, blockIdx.x: %d\n", thread_id, block_sums[blockIdx.x], blockIdx.x);
	}
	__syncthreads();
	if (!has_last_beginning)
	{
		if (threadIdx.x == warps_count * warpSize - 1)
		{
			block_sums[blockIdx.x] = index;
			//printf("thread_id: %d, block_sum: %d, blockIdx.x: %d\n", thread_id, block_sums[blockIdx.x], blockIdx.x);
		}
	}
	grid.sync();

	int block_sum = 0;
	if (thread_id < gridDim.x)
	{
		block_sum = block_sums[thread_id];
	}
	__shared__ UINT partial_sums[2048];	//TODO: change
	grid.sync();
	if (thread_id < gridDim.x)		// maximum is 65535 = 2^16 - 1 threads
	{
		// in-warp scan
		for (int i = 1; i <= warpSize; i *= 2)
		{
			int n = __shfl_up_sync(FULL_MASK, block_sum, i);	// IMPORTANT: what happens when this is not FULL_MASK???

			if (lane_id >= i)
				block_sum += n;
		}

		if (gridDim.x > warpSize)								// if there is more than 32 blocks, there needs to be next level of scan executed
		{
			int lane = warpSize - 1 - __clz(__activemask());
			if (lane_id == lane)
			{
				partial_sums[warp_id] = block_sum;				// last thread in warp writes its warp partial sum to `partial_sums` in sh_mem
			}
		}
	}
	__syncthreads();
	__shared__ UINT partial_sums2[64];
	int partial_sums_len = gridDim.x / warpSize;
	if (gridDim.x % warpSize != 0)
		partial_sums_len++;
	if (thread_id < gridDim.x)
	{
		if (gridDim.x > warpSize)
		{
			if (thread_id < partial_sums_len)			// there is maximum 2048 = 2^11 `partial_sums` values
			{
				int partial_sum = partial_sums[thread_id];

				// in-warp scan
				for (int i = 0; i < warpSize; i *= 2)
				{
					int n = __shfl_up_sync(FULL_MASK, partial_sum, i);

					if (lane_id >= i)
						partial_sum += n;
				}
				// last thread in warp writes its partial sum to memory (another space in shared memory)
				if (partial_sums_len > warpSize)
				{
					int lane = warpSize - 1 - __clz(__activemask());
					if (lane_id == lane)
					{
						partial_sums2[warp_id] = partial_sum;
					}
				}
			}
		}
	}
	__syncthreads();
	__shared__ UINT partial_sums3[2];
	int partial_sums2_len = partial_sums_len / warpSize;
	if (partial_sums_len % warpSize != 0)
		partial_sums2_len++;
	if (gridDim.x > warpSize && thread_id < partial_sums_len)		// there is maximum 64 = 2^6 `partial_sums2` values
	{
		int partial_sum2 = partial_sums2[thread_id];

		// in-warp scan
		for (int i = 0; i < warpSize; i *= 2)
		{
			int n = __shfl_up_sync(FULL_MASK, partial_sum2, i);

			if (lane_id >= i)
				partial_sum2 += n;
		}
		// last thread in warp writes its partial sum to memory (yet another space in shared memory)
		if (partial_sums2_len > warpSize)
		{
			int lane = warpSize - 1 - __clz(__activemask());
			if (lane_id == lane)
			{
				partial_sums3[warp_id] = partial_sum2;
			}
		}
	}
	__syncthreads();
	int partial_sums3_len = partial_sums2_len / warpSize;
	if (partial_sums3_len % warpSize != 0)
		partial_sums3_len++;
	if (partial_sums3_len > 1)
	{
		if (thread_id == 0)
		{
			partial_sums3[1] += partial_sums3[0];
		}
	}
	__syncthreads();
	if (partial_sums2_len > warpSize)
	{
		if (thread_id < partial_sums2_len)
		{
			int ind = thread_id / warpSize;
			if (ind > 0)
				partial_sums2[thread_id] += partial_sums3[ind - 1];
		}
	}
	__syncthreads();
	if (partial_sums_len > warpSize)
	{
		if (thread_id < partial_sums_len)
		{
			int ind = thread_id / warpSize;
			if (ind > 0)
				partial_sums[thread_id] += partial_sums2[ind - 1];
		}
	}
	__syncthreads();
	if (gridDim.x > warpSize)
	{
		if (thread_id < gridDim.x)
		{
			int ind = thread_id / warpSize;
			if (ind > 0)
			{
				//block_sums[thread_id] += partial_sums[ind - 1];
				block_sum += partial_sums[ind - 1];
				//printf("thread_id: %d, block_sum: %d\n", thread_id, block_sum);
			}
		}
	}
	if (thread_id < gridDim.x)
	{
		//printf("thread_id: %d, block_sum: %d\n", thread_id, block_sum);
		block_sums[thread_id] = block_sum;
	}
	//__syncthreads();
	grid.sync();

	if (is_begin)
	{
		int ind = blockIdx.x;
		if (ind > 0)
		{
			//printf("thread %d under ind %d has value: %d\n", thread_id, ind, block_sums[ind - 1]);
			//printf(" has value %d\n", block_sums[ind - 1]);
			int tmp = index;
			index += block_sums[ind - 1];
			//printf("thread %d adds %d to index %d and has %d\n", thread_id, block_sums[ind-1], tmp, index);
		}
	}
	__syncthreads();



	if (is_begin)
	{
		// gather
		//if (warp_id > 0)
		//	index += sums[warp_id - 1];
		unsigned write;
		if (w_type == EMPTY_WORD)
			write = get_compressed(segment_len, 0);
		//output[index - 1] = get_compressed(segment_len, 0);
		else if (w_type == FULL_WORD)
			write = get_compressed(segment_len, 1);
		//output[index - 1] = get_compressed(segment_len, 1);
		else
			write = gulp;
			//output[index - 1] = gulp;
		output[index - 1] = write;
		//printf("thread %d has value: %d\n", thread_id, segment_len);
		//printf("Thread %d writes %u on index %d\n", thread_id, write, index - 1);
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
