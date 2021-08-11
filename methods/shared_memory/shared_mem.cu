#include "cuda_runtime.h"
#include "cooperative_groups.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include "../../bit_functions.cuh"
#include "segment_structs.cuh"
#include "calc_segmentlen_ingrid.cuh"
#include "get_segmentlen_inblock.cuh"
#include "prescan_inblock.cuh"
#include "../../wah_test.h"

#define MAX(a, b) (((a)>(b)) ? (a) : (b))
#define MIN(a, b) (((a)<(b)) ? (a) : (b))

namespace cg = cooperative_groups;


// kernel assumes that grid is 1D
__global__ void SharedMemKernel(UINT* input, int inputSize, UINT* output, size_t* outputSize,
				GET_SEGMENTLEN_FUN get_segmentlen_inblock,
				CALC_SEGMENTLEN_INGRID_FUN calc_segmentlen_ingrid,
				PRESCAN_INBLOCK_FUN prescan_inblock_1,
				PRESCAN_INBLOCK_FUN prescan_inblock_2)
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

	int segment_len = call_get_segmentlen_inblock(get_segmentlen_inblock, is_begin, w_type, smem_ptr, lane_id, warp_id, warps_count);
	// every thread-beginning knows its segment's length (in-block boundaries)

	int index = is_begin ? 1 : 0;
	__syncthreads();

	call_prescan_inblock(prescan_inblock_1, index, smem_ptr, lane_id, warp_id, warps_count);
	// now index is correct in block boundaries

	// ================
	// INTER-BLOCKS STEP
	// ================
	cg::grid_group grid = cg::this_grid();
	call_calc_segmentlen_ingrid(calc_segmentlen_ingrid, segment_len, index, is_begin, w_type, smem_ptr, output, lane_id, warp_id, warps_count, grid);

	// INTER-BLOCKS SCAN
	// write block_sum to global memory
	UINT* g_block_sums = output;		// TODO: possible to test. Allocate memory normally.
	if (threadIdx.x == warps_count * warpSize - 1)
		g_block_sums[blockIdx.x] = index;
	grid.sync();


	// Kernel assumes that there are at least as many threads in a block as the total number of blocks.
	// This assumption makes sense since this kernel is cooperative.
	// Indeed, there ain't many blocks then (usually).
	int block_sum = 0;
	if (thread_id < gridDim.x)
		block_sum = g_block_sums[thread_id];
	grid.sync();

	call_prescan_inblock(prescan_inblock_2, block_sum, smem_ptr, lane_id, warp_id, warps_count);
	if (thread_id < gridDim.x)
		g_block_sums[thread_id] = block_sum;
	grid.sync();

	if (blockIdx.x > 0)
		index += g_block_sums[blockIdx.x - 1];
	grid.sync();

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
	{
		*outputSize = index;
		// printf("outputSize: %d\n", index);
	}
}


// return size of output array if everything successful
// -1 else
long long LaunchKernel(int blocks, int threads_per_block, UINT* d_input, int size, UINT* d_output, size_t* d_outputSize,
	GET_SEGMENTLEN_FUN get_segmentlen_inblock,
	CALC_SEGMENTLEN_INGRID_FUN calc_segmentlen_ingrid,
	PRESCAN_INBLOCK_FUN prescan_inblock_1,
	PRESCAN_INBLOCK_FUN prescan_inblock_2)
{
	if (!isPowerOfTwo(threads_per_block))
	{
		printf("blockSize must be power of 2\n");
		return -1;
	}

	int device = 0;
	cudaDeviceProp deviceProp;
	CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device), Fail);

	int warps_count = threads_per_block / deviceProp.warpSize;											// calc size of needed shared memory (per block)
	if (threads_per_block % deviceProp.warpSize != 0)
		warps_count++;
	size_t smem_size = MAX(MAX(sizeof(segment), sizeof(int)), sizeof(unsigned));
	smem_size = smem_size * threads_per_block;

	int numBlocksPerSm = 0;																				// calc max number of blocks in coop. launch
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, SharedMemKernel, threads_per_block, smem_size), Fail);
	int maxCoopBlocks = deviceProp.multiProcessorCount * numBlocksPerSm;
	maxCoopBlocks = MIN(maxCoopBlocks, threads_per_block);	// there can't be more blocks in kernel launch than thr_p_blcks due to prescan algorithm

	int maxGridSize = maxCoopBlocks * threads_per_block;
	
	size_t outputSize = 0;
	size_t size_left = size;
	int blocks_left = blocks;
	
	UINT* d_outp = d_output;
	UINT* d_inp = d_input;
	void* params[8];
	params[0] = &d_inp;
	params[1] = &maxGridSize;
	params[2] = &d_outp;
	params[3] = &d_outputSize;
	params[4] = &get_segmentlen_inblock;
	params[5] = &calc_segmentlen_ingrid;
	params[6] = &prescan_inblock_1;
	params[7] = &prescan_inblock_2;

	while (blocks_left > maxCoopBlocks)																		// if one coop. launch cannot handle the whole input, handle it in parts
	{
		CUDA_CHECK(cudaLaunchCooperativeKernel((void*)SharedMemKernel, maxCoopBlocks, threads_per_block, params, smem_size), Fail);
		CUDA_CHECK(cudaGetLastError(), Fail);
		CUDA_CHECK(cudaDeviceSynchronize(), Fail);

		size_t oSizeTmp;
		CUDA_CHECK(cudaMemcpy(&oSizeTmp, d_outputSize, sizeof(size_t), cudaMemcpyDeviceToHost), Fail);
		d_inp += maxGridSize;
		d_outp += oSizeTmp;
		outputSize += oSizeTmp;

		blocks_left -= maxCoopBlocks;
		size_left -= maxGridSize;
	}
	if (blocks_left > 0)																				// handle the rest of input
	{
		params[1] = &size_left;
		CUDA_CHECK(cudaLaunchCooperativeKernel((void*)SharedMemKernel, blocks_left, threads_per_block, params, smem_size), Fail);
		CUDA_CHECK(cudaGetLastError(), Fail);
		CUDA_CHECK(cudaDeviceSynchronize(), Fail);

		size_t oSizeTmp;
		CUDA_CHECK(cudaMemcpy(&oSizeTmp, d_outputSize, sizeof(size_t), cudaMemcpyDeviceToHost), Fail);
		outputSize += oSizeTmp;
	}

	int blcks = outputSize / threads_per_block;
	if (outputSize % threads_per_block != 0)
		blcks++;
	ballot_warp_merge<<<blcks, threads_per_block>>>(outputSize, d_output, d_input);						// join parts
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

UINT* SharedMemWAHParameterized(int size, UINT* d_input, int __threads_per_block,
	GET_SEGMENTLEN_FUN get_segmentlen_inblock,
	CALC_SEGMENTLEN_INGRID_FUN calc_segmentlen_ingrid,
	PRESCAN_INBLOCK_FUN prescan_inblock_1,
	PRESCAN_INBLOCK_FUN prescan_inblock_2)
{
	if (!ensure_cooperativity_support())
		return nullptr;

	UINT* d_output;
	size_t* d_outputSize;
	CUDA_CHECK(cudaMalloc((void**)&d_output, size * sizeof(UINT)), Fin);
	CUDA_CHECK(cudaMalloc((void**)&d_outputSize, sizeof(size_t)), FreeOutput);

	#ifdef BLOCKSIZE
	int threads_per_block = BLOCKSIZE;
	#else
	int threads_per_block = 1024;	// TODO: change to __threads_per_block
	#endif

	int blocks = size / threads_per_block;
	if (size % threads_per_block != 0)
		blocks++;
	
	long long outputSize = LaunchKernel(blocks, threads_per_block, d_input, size, d_output, d_outputSize,
							get_segmentlen_inblock, calc_segmentlen_ingrid, prescan_inblock_1, prescan_inblock_2);

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
Fin:
	return result;
}



GET_SEGMENTLEN_FUN get_segmentlen_inblock_fun = GET_SEGMENTLEN_INBLOCK_SOA;
CALC_SEGMENTLEN_INGRID_FUN calc_segmentlen_ingrid_fun = CALC_SEGMENTLEN_INGRID_COALESCED;
PRESCAN_INBLOCK_FUN prescan_inblock_1_fun = PRESCAN_SHFL;
PRESCAN_INBLOCK_FUN prescan_inblock_2_fun = PRESCAN_SHFL;

UINT* SharedMemWAH(int size, UINT* d_input, int __threads_per_block)
{
	return SharedMemWAHParameterized(size, d_input, __threads_per_block, get_segmentlen_inblock_fun, calc_segmentlen_ingrid_fun, prescan_inblock_1_fun, prescan_inblock_2_fun);
}

void smem_iterate_unittests()
{
	for (int get_segmentlen_inblock = GET_SEGMENTLEN_INBLOCK_AOS; get_segmentlen_inblock <= GET_SEGMENTLEN_INBLOCK_SOA; get_segmentlen_inblock++)
	{
		for (int calc_segmentlen_ingrid = CALC_SEGMENTLEN_INGRID_NONCOALESCED; calc_segmentlen_ingrid <= CALC_SEGMENTLEN_INGRID_COALESCED; calc_segmentlen_ingrid++)
		{
			#ifdef UNROLLED_PRESCANS
			for (int prescan_inblock_1 = PRESCAN_SHFL; prescan_inblock_1 <= PRESCAN_WORKEFFICIENT_CONFLICTFREE_UNROLLED; prescan_inblock_1++)
			#else
			for (int prescan_inblock_1 = PRESCAN_SHFL; prescan_inblock_1 <= PRESCAN_WORKEFFICIENT_CONFLICTFREE; prescan_inblock_1++)
			#endif
			{
				#ifdef UNROLLED_PRESCANS
				for (int prescan_inblock_2 = PRESCAN_SHFL; prescan_inblock_2 <= PRESCAN_WORKEFFICIENT_CONFLICTFREE_UNROLLED; prescan_inblock_2++)
				#else
				for (int prescan_inblock_2 = PRESCAN_SHFL; prescan_inblock_2 <= PRESCAN_WORKEFFICIENT_CONFLICTFREE; prescan_inblock_2++)
				#endif
				{
					get_segmentlen_inblock_fun = (GET_SEGMENTLEN_FUN)get_segmentlen_inblock;
					calc_segmentlen_ingrid_fun = (CALC_SEGMENTLEN_INGRID_FUN)calc_segmentlen_ingrid;
					prescan_inblock_1_fun = (PRESCAN_INBLOCK_FUN)prescan_inblock_1;
					prescan_inblock_2_fun = (PRESCAN_INBLOCK_FUN)prescan_inblock_2;

					UnitTests(&SharedMemWAH);
				}
			}
		}
	}
}

void smem_iterate_benchmark(int batch_reserve, int batch_pos, int batch_size, int threads_per_block, std::string data_filename, UINT* data)
{
	int prescan_inblock_enum[3] = {PRESCAN_SHFL, PRESCAN_WORKEFFICIENT, PRESCAN_WORKEFFICIENT_CONFLICTFREE};
	std::string prescan_inblock_name[3] = {"prescan_shuffle_","prescan_workefficient_","prescan_workefficient_conflict_free_"};

	int get_segmentlen_inblock_enum[3] = {GET_SEGMENTLEN_INBLOCK_AOS, GET_SEGMENTLEN_INBLOCK_SOA};
	std::string get_segmentlen_inblock_name[3] = {"AOS_","SOA_"};

	int calc_segmentlen_ingrid_enum[2] = {CALC_SEGMENTLEN_INGRID_NONCOALESCED, CALC_SEGMENTLEN_INGRID_COALESCED};
	std::string calc_segmentlen_ingrid_name[2] = {"noncoalesced_","coalesced_"};


	UINT* d_data;
    cudaMalloc((UINT**)&d_data, sizeof(UINT) * batch_reserve);


	for (int get_segmentlen_inblock = 0; get_segmentlen_inblock <= 1; get_segmentlen_inblock++)
	{
		for (int calc_segmentlen_ingrid = 0; calc_segmentlen_ingrid <= 1; calc_segmentlen_ingrid++)
		{
			for (int prescan_inblock_1 = 0; prescan_inblock_1 <= 2; prescan_inblock_1++)
			{
				for (int prescan_inblock_2 = 0; prescan_inblock_2 <= 1; prescan_inblock_2++)
				{
					get_segmentlen_inblock_fun = (GET_SEGMENTLEN_FUN)get_segmentlen_inblock;
					calc_segmentlen_ingrid_fun = (CALC_SEGMENTLEN_INGRID_FUN)calc_segmentlen_ingrid;
					prescan_inblock_1_fun = (PRESCAN_INBLOCK_FUN)prescan_inblock_1;
					prescan_inblock_2_fun = (PRESCAN_INBLOCK_FUN)prescan_inblock_2;

					std::string method_name = "sharedmem__"+prescan_inblock_name[prescan_inblock_1]+prescan_inblock_name[prescan_inblock_2]+get_segmentlen_inblock_name[get_segmentlen_inblock]+calc_segmentlen_ingrid_name[calc_segmentlen_ingrid]+";";
					cudaMemcpy(d_data, data, sizeof(UINT) * batch_reserve, cudaMemcpyHostToDevice);
					Benchmark(&SharedMemWAH, batch_reserve, d_data, data_filename + ";" + std::to_string(batch_pos) + ";" + std::to_string(batch_reserve * 32) + ";" + method_name + std::to_string(threads_per_block) + ";", threads_per_block);
				}
			}
		}
	}

	cudaFree(d_data);
}

void unrolled_smem_iterate_benchmark(int batch_reserve, int batch_pos, int batch_size, int threads_per_block, std::string data_filename, UINT* data)
{
	int prescan_inblock_enum[3] = {PRESCAN_SHFL_UNROLLED, PRESCAN_WORKEFFICIENT_UNROLLED, PRESCAN_WORKEFFICIENT_CONFLICTFREE_UNROLLED};
	std::string prescan_inblock_name[3] = {"unrolled_prescan_shuffle_","unrolled_prescan_workefficient_","unrolled_prescan_workefficient_conflict_free_"};

	int get_segmentlen_inblock_enum[3] = {GET_SEGMENTLEN_INBLOCK_AOS, GET_SEGMENTLEN_INBLOCK_SOA};
	std::string get_segmentlen_inblock_name[3] = {"AOS_","SOA_"};

	int calc_segmentlen_ingrid_enum[2] = {CALC_SEGMENTLEN_INGRID_NONCOALESCED, CALC_SEGMENTLEN_INGRID_COALESCED};
	std::string calc_segmentlen_ingrid_name[2] = {"noncoalesced_","coalesced_"};


	UINT* d_data;
    cudaMalloc((UINT**)&d_data, sizeof(UINT) * batch_reserve);


	for (int get_segmentlen_inblock = 0; get_segmentlen_inblock <= 1; get_segmentlen_inblock++)
	{
		for (int calc_segmentlen_ingrid = 0; calc_segmentlen_ingrid <= 1; calc_segmentlen_ingrid++)
		{
			for (int prescan_inblock_1 = 0; prescan_inblock_1 <= 2; prescan_inblock_1++)
			{
				for (int prescan_inblock_2 = 0; prescan_inblock_2 <= 1; prescan_inblock_2++)
				{
					get_segmentlen_inblock_fun = (GET_SEGMENTLEN_FUN)get_segmentlen_inblock;
					calc_segmentlen_ingrid_fun = (CALC_SEGMENTLEN_INGRID_FUN)calc_segmentlen_ingrid;
					prescan_inblock_1_fun = (PRESCAN_INBLOCK_FUN)prescan_inblock_1;
					prescan_inblock_2_fun = (PRESCAN_INBLOCK_FUN)prescan_inblock_2;

					std::string method_name = "sharedmem__"+prescan_inblock_name[prescan_inblock_1]+prescan_inblock_name[prescan_inblock_2]+get_segmentlen_inblock_name[get_segmentlen_inblock]+calc_segmentlen_ingrid_name[calc_segmentlen_ingrid]+";";
					cudaMemcpy(d_data, data, sizeof(UINT) * batch_reserve, cudaMemcpyHostToDevice);
					Benchmark(&SharedMemWAH, batch_reserve, d_data, data_filename + ";" + std::to_string(batch_pos) + ";" + std::to_string(batch_reserve * 32) + ";" + method_name + std::to_string(threads_per_block) + ";", threads_per_block);
				}
			}
		}
	}

	cudaFree(d_data);
}
