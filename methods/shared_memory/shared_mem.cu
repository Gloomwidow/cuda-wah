#include "cuda_runtime.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cooperative_groups.h>
// #include "get_segmentlen_inblock.cu"
// #include "scan_inblock.cu"
// #include "calc_segmentlen_ingrid.cu"
#include "smem_functions.cuh"
#include "../../wah_test.h"	// TODO: usunąć
namespace cg = cooperative_groups;

#ifndef WAH_FUN_TYPEDEF
#define WAH_FUN_TYPEDEF
typedef UINT* (*WAH_fun)(int size, UINT* input);
#endif // WAH_FUN_TYPEDEF

#define MAX(a,b) (((a)>(b)) ? (a) : (b))


typedef bool (*calc_segmentlen_ingrid_fun)(int& segment_len, int& index, bool& is_begin, WORD_TYPE w_type, void* smem_ptr, void* gmem_ptr, int lane_id, int warp_id, int warps_count, cg::grid_group grid);

__device__ bool calc_segmentlen_ingrid_noncoalesced_sync(int &segment_len, int &index, bool &is_begin, WORD_TYPE w_type, void* smem_ptr, void* gmem_ptr, int lane_id, int warp_id, int warps_count, cg::grid_group grid)
{
	segment* g_block_segments = (segment*)gmem_ptr;														// global memory allocation is just being reused
	unsigned* s_block_begins_masks = (unsigned*)smem_ptr;

	// ==== find the last thread-beginning in block ====
	unsigned warp_begins_mask = __ballot_sync(FULL_MASK, is_begin);
	if (lane_id == 0)
		s_block_begins_masks[warp_id] = warp_begins_mask;
	__syncthreads();

	if (threadIdx.x < warps_count)																		// find last warp in block which contains any thread-beginning
	{
		unsigned begins_mask = s_block_begins_masks[threadIdx.x];
		unsigned is_mask_nonzero = __ballot_sync(__activemask(), begins_mask != EMPTY_MASK);
		if (threadIdx.x == 0)
			s_block_begins_masks[0] = (unsigned)(warpSize - 1 - __clz(is_mask_nonzero));				// write its warp_id in shared memory
	}
	__syncthreads();

	// ==== write block_segments in global memory ====
	bool am_last_beginning_inblock = false;
	if (warp_id == s_block_begins_masks[0])																// now we find last thread-beginning in block (in previously found warp)
	{																									// and write it's segment info in global memory
		int lane = warpSize - 1 - __clz(warp_begins_mask);	// lane_id of this thread
		if (lane_id == lane)
		{
			am_last_beginning_inblock = true;

			g_block_segments[blockIdx.x].r_end_type = w_type;
			g_block_segments[blockIdx.x].r_end_len = segment_len;
		}
	}
	if (threadIdx.x == 0)						// first thread-beginning in block
	{
		g_block_segments[blockIdx.x].l_end_type = w_type;
		g_block_segments[blockIdx.x].l_end_len = segment_len;
	}
	grid.sync();

	// ==== update segment_len and is_begin so that it's correct in grid boundaries ====
	bool* s_decrement_index = (bool*)smem_ptr;
	if (threadIdx.x == 0)
		*s_decrement_index = false;
	__syncthreads();
	if (is_begin)
	{
		if (blockIdx.x > 0 && threadIdx.x == 0 && w_type != TAIL_WORD &&								// check if the first thread-beginning in block is really
			g_block_segments[blockIdx.x - 1].r_end_type == w_type)										// thread-beginning in the context of the grid...
		{
			is_begin = false;
			am_last_beginning_inblock = false;
			*s_decrement_index = true;
		}

		if (am_last_beginning_inblock)																	// ...if not, the last thread-beginning form prev. block should add sth to its `segment_len`
		{
			for (int i = blockIdx.x + 1; i < gridDim.x && g_block_segments[i].l_end_type == w_type; i++)
			{
				segment_len += g_block_segments[i].l_end_len;		// check types
				if (g_block_segments[i].l_end_len != blockDim.x)
					break;
			}
		}
	}
	__syncthreads();
	if (*s_decrement_index)																				// if the first thread is no longer thread-beginning,
		index -= 1;																			            // whole block needs to decrement index

	grid.sync();
	return am_last_beginning_inblock;
}

// __device__ bool calc_segmentlen_ingrid_coalesced_sync(int &segment_len, int &index, bool &is_begin, WORD_TYPE w_type, void* smem_ptr, void* gmem_ptr, int lane_id, int warp_id, int warps_count, cg::grid_group grid)
// {
// 	segment_soa g_block_segments;
//     int stride;
//     if (gridDim.x % 32 != 0)
//         stride = ((gridDim.x / 32) + 1) * 32;
//     else
//         stride = gridDim.x;
//     g_block_segments.l_end_type = (WORD_TYPE*)gmem_ptr;												    // global memory allocation is just being reused
//     g_block_segments.l_end_len = (int*)(g_block_segments.l_end_type[stride]);
//     g_block_segments.r_end_type = (WORD_TYPE*)(g_block_segments.l_end_len[stride]);
//     g_block_segments.r_end_len = (int*)(g_block_segments.r_end_type[stride]);

// 	unsigned* s_block_begins_masks = (unsigned*)smem_ptr;

// 	// ==== find the last thread-beginning in block ====
// 	unsigned warp_begins_mask = __ballot_sync(FULL_MASK, is_begin);
// 	if (lane_id == 0)
// 		s_block_begins_masks[warp_id] = warp_begins_mask;
// 	__syncthreads();

// 	if (threadIdx.x < warps_count)																		// find last warp in block which contains any thread-beginning
// 	{
// 		unsigned begins_mask = s_block_begins_masks[threadIdx.x];
// 		unsigned is_mask_nonzero = __ballot_sync(__activemask(), begins_mask != EMPTY_MASK);
// 		if (threadIdx.x == 0)
// 			s_block_begins_masks[0] = (unsigned)(warpSize - 1 - __clz(is_mask_nonzero));				// write its warp_id in shared memory
// 	}
// 	__syncthreads();

// 	// ==== write block_segments in global memory ====
// 	bool am_last_beginning_inblock = false;
// 	if (warp_id == s_block_begins_masks[0])																// now we find last thread-beginning in block (in previously found warp)
// 	{																									// and write it's segment info in global memory
// 		int lane = warpSize - 1 - __clz(warp_begins_mask);	// lane_id of this thread
// 		if (lane_id == lane)
// 		{
// 			am_last_beginning_inblock = true;

// 			g_block_segments.r_end_type[blockIdx.x] = w_type;
// 			g_block_segments.r_end_len[blockIdx.x] = segment_len;
// 		}
// 	}
// 	if (threadIdx.x == 0)						// first thread-beginning in block
// 	{
// 		g_block_segments.l_end_type[blockIdx.x] = w_type;
// 		g_block_segments.l_end_len[blockIdx.x] = segment_len;
// 	}
// 	grid.sync();

// 	// ==== update segment_len and is_begin so that it's correct in grid boundaries ====
// 	bool* s_decrement_index = (bool*)smem_ptr;
// 	if (threadIdx.x == 0)
// 		*s_decrement_index = false;
// 	__syncthreads();
// 	if (is_begin)
// 	{
// 		if (blockIdx.x > 0 && threadIdx.x == 0 && w_type != TAIL_WORD &&								// check if the first thread-beginning in block is really
// 			g_block_segments.r_end_type[blockIdx.x] == w_type)  										// thread-beginning in the context of the grid...
// 		{
// 			is_begin = false;
// 			am_last_beginning_inblock = false;
// 			*s_decrement_index = true;
// 		}

// 		if (am_last_beginning_inblock)																	// ...if not, the last thread-beginning form prev. block should add sth to its `segment_len`
// 		{
// 			for (int i = blockIdx.x + 1; i < gridDim.x && g_block_segments.l_end_type[i] == w_type; i++)
// 			{
// 				segment_len += g_block_segments.l_end_len[i];
// 				if (g_block_segments.l_end_len[i] != blockDim.x)
// 					break;
// 			}
// 		}
// 	}
// 	__syncthreads();
// 	if (*s_decrement_index)																				// if the first thread is no longer thread-beginning,
// 		index -= 1;																			            // whole block needs to decrement index

// 	grid.sync();
// 	return am_last_beginning_inblock;
// }










// kernel assumes that grid is 1D
template<get_segmentlen_inblock_fun get_segmentlen_inblock,
		inclusive_scan_inblock_fun inclusive_scan_inblock,
		calc_segmentlen_ingrid_fun calc_segmentlen_ingrid
		>
__global__ void SharedMemKernel(UINT* input, int inputSize, UINT* output, size_t* outputSize)
{
	extern __shared__ int smem_ptr[];

	const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	const int lane_id = threadIdx.x % warpSize;
	const int warp_id = threadIdx.x / warpSize;
	int warps_count;
	if ((blockIdx.x + 1) * blockDim.x > inputSize)	// last block can enter here
	{
		warps_count = (inputSize - blockIdx.x * blockDim.x) / warpSize;
		if (inputSize % warpSize != 0)
			warps_count++;
	}
	else
		warps_count = blockDim.x / warpSize;

	UINT gulp;
	if (thread_id < inputSize)
		gulp = input[thread_id];
	WORD_TYPE w_type = get_word_type(gulp);

	bool is_begin; 	// is this thread the beginning of a section?
	char prev_type = __shfl_up_sync(FULL_MASK, w_type, 1);
	if (thread_id < inputSize)
	{
		if (lane_id == 0)
			is_begin = true;
		else
			is_begin = (w_type == TAIL_WORD) || (w_type != prev_type);
	}
	else
		is_begin = false;

	int segment_len = get_segmentlen_inblock(is_begin, w_type, smem_ptr, lane_id, warp_id, warps_count);
	// int segment_len = get_segmentlen_inblock_aos_sync(&is_begin, w_type, smem_ptr, lane_id, warp_id, warps_count);
	// every thread-beginning knows its segment's length (in-block boundaries)

	int index = is_begin ? 1 : 0;
	__syncthreads();/////
	inclusive_scan_inblock(index, smem_ptr, lane_id, warp_id, warps_count);
	// now index is correct in block boundaries


	// INTER-BLOCKS STEP
	cg::grid_group grid = cg::this_grid();
	bool am_last_beginning_inblock = calc_segmentlen_ingrid(segment_len, index, is_begin, w_type, smem_ptr, output, lane_id, warp_id, warps_count, grid);
	
	// INTER-BLOCKS SCAN
	UINT* g_block_sums = output;		// TODO: possible to test. Allocate memory normally.

	bool* has_last_beginning = (bool*)smem_ptr;															// write block_sum (index) to global memory
	if (threadIdx.x == 0)
		*has_last_beginning = false;
	__syncthreads();
	if (am_last_beginning_inblock)
	{
		*has_last_beginning = true;
		g_block_sums[blockIdx.x] = index;
	}
	__syncthreads();
	if (!(*has_last_beginning))
	{
		if (threadIdx.x == warps_count * warpSize - 1)
			g_block_sums[blockIdx.x] = index;
	}
	if (threadIdx.x == warps_count * warpSize - 1)															// last thread in block writes block_sum to global memory
		g_block_sums[blockIdx.x] = index;
	grid.sync();


	// Kernel assumes that there are at least as many threads_in_block as the total number of blocks.
	// This assumption makes sense since this kernel is cooperative.
	// Indeed, there ain't many blocks then (usually).
	int block_sum = 0;
	if (thread_id < gridDim.x)
		block_sum = g_block_sums[thread_id];
	grid.sync();

	inclusive_scan_inblock(block_sum, smem_ptr, lane_id, warp_id, warps_count);
	if (thread_id < gridDim.x)
		g_block_sums[thread_id] = block_sum;
	grid.sync();

	if (blockIdx.x > 0)
		index += g_block_sums[blockIdx.x - 1];
	grid.sync();

	// WRITE COMPRESSED
	if (is_begin)
	{
		if (w_type == EMPTY_WORD)
			output[index - 1] = get_compressed(segment_len, 0);
		else if (w_type == FULL_WORD)
			output[index - 1] = get_compressed(segment_len, 1);
		else
			output[index - 1] = gulp;
	}
	if (thread_id == inputSize - 1)
		*outputSize = index;
}

// return size of output array if everything successful
// -1 else
template<get_segmentlen_inblock_fun get_segmentlen_inblock,
		inclusive_scan_inblock_fun inclusive_scan_inblock,
		calc_segmentlen_ingrid_fun calc_segmentlen_ingrid
		>
long long LaunchKernel(int blocks, int threads_per_block,
						UINT* d_input, int size, UINT* d_output, size_t* d_outputSize
						// const get_segmentlen_inblock_fun get_segmentlen_inblock,
						// const inclusive_scan_inblock_fun inclusive_scan_inblock,
						// const calc_segmentlen_ingrid_fun calc_segmentlen_ingrid
					)
{
	int device = 0;
	cudaDeviceProp deviceProp;
	CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device), Fail);

	int warps_count = threads_per_block / deviceProp.warpSize;											// calc size of needed shared memory (per block)
	if (threads_per_block % deviceProp.warpSize != 0)
		warps_count++;
	size_t smem_size = MAX(MAX(sizeof(segment), sizeof(int)), sizeof(unsigned));
	smem_size = smem_size * warps_count;

	int numBlocksPerSm = 0;																				// calc max number of blocks in coop. launch
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, SharedMemKernel<get_segmentlen_inblock, inclusive_scan_inblock, calc_segmentlen_ingrid>,
															threads_per_block, smem_size), Fail);
	int maxCoopBlocks = deviceProp.multiProcessorCount * numBlocksPerSm;
	// printf("needed blocks: %d, max blocks: %d\n", blocks, maxCoopBlocks);

	if (blocks > maxCoopBlocks && threads_per_block < maxCoopBlocks)
	{
		printf("insufficient number of threads_per_block to make cooperative scan on whole grid\n");	// this can only happen when GPU has very many of SMs
		return -1;																						// (or more precisely: cooperative launch can handle many blocks)
	}																									// and blockSize is smaller than that

	int maxGridSize = maxCoopBlocks * threads_per_block;
	void* params[4];

	size_t outputSize = 0;
	size_t size_left = size;
	int blocks_left = blocks;

	UINT* d_outp = d_output;
	UINT* d_inp = d_input;
	params[0] = &d_inp;
	params[1] = &maxGridSize;
	params[2] = &d_outp;
	params[3] = &d_outputSize;

	while (blocks_left > maxCoopBlocks)																	// if one coop. launch cannot handle the whole input, handle it in parts
	{
		CUDA_CHECK(cudaLaunchCooperativeKernel((void*)SharedMemKernel<get_segmentlen_inblock, inclusive_scan_inblock, calc_segmentlen_ingrid>, maxCoopBlocks, threads_per_block, params, smem_size), Fail);
		CUDA_CHECK(cudaGetLastError(), Fail);
		CUDA_CHECK(cudaDeviceSynchronize(), Fail);

		size_t oSizeTmp;
		CUDA_CHECK(cudaMemcpy(&oSizeTmp, d_outputSize, sizeof(size_t), cudaMemcpyDeviceToHost), Fail);
		// CUDA_CHECK(cudaMemcpy(outp_curr_ptr, *(params[2]), outputSize * sizeof(UINT), cudaMemcpyDeviceToHost), Fail);
		d_inp += maxGridSize;
		d_outp += oSizeTmp;
		outputSize += oSizeTmp;

		blocks_left -= maxCoopBlocks;
		size_left -= maxGridSize;
	}
	if (blocks_left > 0)																				// handle the rest of input
	{
		params[1] = &size_left;
		CUDA_CHECK(cudaLaunchCooperativeKernel((void*)SharedMemKernel<get_segmentlen_inblock, inclusive_scan_inblock, calc_segmentlen_ingrid>, blocks_left, threads_per_block, params, smem_size), Fail);
		CUDA_CHECK(cudaGetLastError(), Fail);
		CUDA_CHECK(cudaDeviceSynchronize(), Fail);

		size_t oSizeTmp;
		CUDA_CHECK(cudaMemcpy(&oSizeTmp, d_outputSize, sizeof(size_t), cudaMemcpyDeviceToHost), Fail);
		outputSize += oSizeTmp;
	}

	int threads_p_block = 512;
	int blcks = outputSize / threads_p_block;
	if (outputSize % threads_p_block != 0)
		blcks++;
	ballot_warp_merge << <blcks, threads_p_block >> > (outputSize, d_output, d_input);					// join parts
	CUDA_CHECK(cudaGetLastError(), Fail);
	CUDA_CHECK(cudaDeviceSynchronize(), Fail);

	UINT* final_end = thrust::remove_if(thrust::device, d_input, d_input + outputSize, wah_zero());		// remove leftover gaps
	int final_count = final_end - d_input;

	return final_count;

Fail:
	return -1;
}


bool ensure_cooperativity_support()
{
	cudaDeviceProp deviceProp = { 0 };

	int device;
	CUDA_CHECK(cudaGetDevice(&device), Finish);

	CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device), Finish);
	if (!deviceProp.cooperativeLaunch)
	{
		printf("\nSelected GPU (%d) does not support Cooperative Kernel Launch, Waiving the run\n", device);
		return false;
	}
	return true;

Finish:
	return false;
}

template<get_segmentlen_inblock_fun get_segmentlen_inblock,
		inclusive_scan_inblock_fun inclusive_scan_inblock,
		calc_segmentlen_ingrid_fun calc_segmentlen_ingrid
		>
UINT* SharedMemWAH(int size, UINT* d_input, int threads_per_b)
{
	if (size < 1 || d_input == nullptr)
	{
		printf("bad argument\n");
		return nullptr;
	}
	if (!ensure_cooperativity_support())
		return nullptr;

	// UINT* d_input;
	UINT* d_output;
	size_t* d_outputSize;
	CUDA_CHECK(cudaMalloc((void**)&d_input, size * sizeof(UINT)), Fin);		// reinterpret_cast<>
	CUDA_CHECK(cudaMalloc((void**)&d_output, size * sizeof(UINT)), Fin);
	CUDA_CHECK(cudaMalloc((void**)&d_outputSize, sizeof(size_t)), FreeOutput);
	// CUDA_CHECK(cudaMemcpy(d_input, input, size * sizeof(UINT), cudaMemcpyHostToDevice), Free);

	int threads_per_block = 1024;
	int blocks = size / threads_per_block;
	if (size % threads_per_block != 0)
		blocks++;

	// SharedMemKernel<<<blocks, threads_per_block>>>(d_input, size, d_output, d_outputSize);
	long long outputSize = LaunchKernel<get_segmentlen_inblock,
										inclusive_scan_inblock,
										calc_segmentlen_ingrid>(blocks, threads_per_block, d_input, size, d_output, d_outputSize);

	if (outputSize < 0)
	{
		printf("something went wrong\n");
		goto Free;
	}
	UINT* result = new UINT[outputSize];
	CUDA_CHECK(cudaMemcpy(result, d_input, outputSize * sizeof(UINT), cudaMemcpyDeviceToHost), Free);

Free:
	CUDA_CHECK(cudaFree(d_outputSize), FreeOutput);
FreeOutput:
	CUDA_CHECK(cudaFree(d_output), Fin);
// FreeInput:
// 	CUDA_CHECK(cudaFree(d_input), Fin);
Fin:
	return result;
}

WAH_fun* get_wahs(int* count)
{
	// get_segmentlen_inblock_fun* segmentlen_inblocks = new get_segmentlen_inblock_fun[2];
	// get_segmentlen_inblock_fun* segmentlen_inblocks = new get_segmentlen_inblock_fun[2] { get_segmentlen_inblock_soa_sync, get_segmentlen_inblock_aos_sync };
	// inclusive_scan_inblock_fun* scan_inblocks = new inclusive_scan_inblock_fun[6] { inclusive_scan_inblock_shfl_sync, scan_inblock_workefficient_sync, scan_inblock_workefficient_noconflict_sync,
	// 						inclusive_scan_inblock_shfl_unrolled_sync, scan_inblock_workefficient_unrolled_sync, scan_inblock_workefficient_noconflict_unrolled_sync };
	// calc_segmentlen_ingrid_fun* segmentlen_ingrids = new calc_segmentlen_ingrid_fun[2] { calc_segmentlen_ingrid_noncoalesced_sync, calc_segmentlen_ingrid_coalesced_sync };

	WAH_fun* wahs = new WAH_fun[24];
	// for (int i = 0; i < 2; i++)
	// {
	// 	for (int j = 0; j < 6; j++)
	// 	{
	// 		for (int k = 0; k < 2; k++)
	// 		{
	// 			wahs[i*j*k] = SharedMemWAH<segmentlen_inblocks[i],
	// 										scan_inblocks[j],
	// 										segmentlen_ingrids[k]>;
	// 		}
	// 	}
	// }
	
	*count = 2*2*6;
	return wahs;
	// SharedMemWAH<get_segmentlen_inblock_aos_sync, inclusive_scan_inblock_shfl_sync, calc_segmentlen_ingrid_noncoalesced_sync>(size, input);
}

void run()
{
	UnitTests(&SharedMemWAH<get_segmentlen_inblock_aos_sync, inclusive_scan_inblock_shfl_sync, calc_segmentlen_ingrid_noncoalesced_sync>);
}
