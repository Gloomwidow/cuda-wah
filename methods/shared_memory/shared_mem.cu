#include "cuda_runtime.h"
#include "cooperative_groups.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "../../bit_functions.cuh"
#include "get_segmentlen_inblock.cu"
#include "scan_inblock.cu"
#include "calc_segmentlen_inblock.cu"

#define MAX(a,b) (((a)>(b)) ? (a) : (b))

namespace cg = cooperative_groups;

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

	int segment_len = get_segmentlen_inblock(&is_begin, w_type, smem_ptr, lane_id, warp_id, warps_count);
	// int segment_len = get_segmentlen_inblock_aos_sync(&is_begin, w_type, smem_ptr, lane_id, warp_id, warps_count);
	// every thread-beginning knows its segment's length (in-block boundaries)

	int index = is_begin ? 1 : 0;
	__syncthreads();/////
	inclusive_scan_inblock_sync(&index, smem_ptr, lane_id, warp_id, warps_count);
	// now index is correct in block boundaries


	// INTER-BLOCKS STEP
	cg::grid_group grid = cg::this_grid();
	/*bool am_last_beginning_inblock = */calc_segmentlen_ingrid(&segment_len, &index, &is_begin, w_type, smem_ptr, output, lane_id, warp_id, warps_count, grid);
	
	// INTER-BLOCKS SCAN
	UINT* g_block_sums = output;		// TODO: possible to test. Allocate memory normally.

	// bool* has_last_beginning = (bool*)smem_ptr;															// write block_sum (index) to global memory
	// if (threadIdx.x == 0)
	// 	*has_last_beginning = false;
	// __syncthreads();
	// if (am_last_beginning_inblock)
	// {
	// 	*has_last_beginning = true;
	// 	g_block_sums[blockIdx.x] = index;
	// }
	// __syncthreads();
	// if (!(*has_last_beginning))
	// {
	// 	if (threadIdx.x == warps_count * warpSize - 1)
	// 		g_block_sums[blockIdx.x] = index;
	// }
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

	inclusive_scan_inblock_sync(&block_sum, smem_ptr, lane_id, warp_id, warps_count);
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
long long LaunchKernel(int blocks, int threads_per_block,
						UINT* d_input, int size, UINT* d_output, size_t* d_outputSize,
						get_segmentlen_inblock_fun get_segmentlen_inblock,
						inclusive_scan_inblock_fun inclusive_scan_inblock,
						calc_segmentlen_ingrid_fun calc_segmentlen_ingrid
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
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, SharedMemKernel, threads_per_block, smem_size), Fail);
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
UINT* SharedMemWAH(int size, UINT* d_input)
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
	CUDA_CHECK(cudaMalloc((void**)&d_output, size * sizeof(UINT)), FreeInput);
	CUDA_CHECK(cudaMalloc((void**)&d_outputSize, sizeof(size_t)), FreeOutput);
	// CUDA_CHECK(cudaMemcpy(d_input, input, size * sizeof(UINT), cudaMemcpyHostToDevice), Free);

	int threads_per_block = 1024;
	int blocks = size / threads_per_block;
	if (size % threads_per_block != 0)
		blocks++;

	// SharedMemKernel<<<blocks, threads_per_block>>>(d_input, size, d_output, d_outputSize);
	long long outputSize = LaunchKernel(blocks, threads_per_block, d_input, size, d_output, d_outputSize,									
										get_segmentlen_inblock,
										inclusive_scan_inblock_fun inclusive_scan_inblock,
										calc_segmentlen_ingrid
							);

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
	CUDA_CHECK(cudaFree(d_output), FreeInput);
// FreeInput:
// 	CUDA_CHECK(cudaFree(d_input), Fin);
Fin:
	return result;
}

WAH_fun* get_wahs(int* count)
{
	get_segmentlen_inblock_fun* segmentlen_inblocks = new get_segmentlen_inblock_fun[2];
	inclusive_scan_inblock_fun* scan_inblocks = new inclusive_scan_inblock_fun[6];
	calc_segmentlen_ingrid_fun* segmentlen_ingrids = new calc_segmentlen_ingrid_fun[2];

	WAH_fun* wahs = new WAH_fun[24];
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			for (int k = 0; k < 2; k++)
			{
				wahs[i*j*k] = SharedMemWAH<segmentlen_inblocks[i]
											scan_inblocks[j],
											segmentlen_ingrids[k]>;
			}
		}
	}
	
	*count = 2*2*6;
	return wahs;
	// SharedMemWAH<get_segmentlen_inblock_aos_sync, inclusive_scan_inblock_shfl_sync, calc_segmentlen_ingrid_noncoalesced_sync>(size, input);
}
