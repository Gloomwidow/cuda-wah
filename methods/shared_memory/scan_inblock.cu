#include "smem_functions.cuh"
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) \
	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

#define WARPSIZE 32
#define WARPSCOUNT 32
#define BLOCKSIZE 1024

// taken from https://github.com/NVIDIA/cuda-samples/blob/master/Samples/shfl_scan/shfl_scan.cu
__device__ void inclusive_scan_inblock_shfl_sync(int &value, void* smem_ptr, int lane_id, int warp_id, int warps_count)
{
	int* s_sums = (int*)smem_ptr;

	// in-warp scan
	for (int i = 1; i <= warpSize; i *= 2)
	{
		int n = __shfl_up_sync(FULL_MASK, value, i);	// add width as argument???

		if (lane_id >= i)
			value += n;
	}
	if (warps_count == 1)
		return;

	// inter-warp scan
	if (lane_id == warpSize - 1)
		s_sums[warp_id] = value;
	__syncthreads();

	// the same shfl scan operation, but performed on warp sums
	// this can be safely done by a single warp, since there is maximum of 32 warps in a block
	if (warp_id == 0 && lane_id < warps_count)
	{
		int warp_sum = s_sums[lane_id];

		int mask = (1 << warps_count) - 1;
		for (int i = 1; i <= warps_count; i *= 2)
		{
			int n = __shfl_up_sync(mask, warp_sum, i);
			if (lane_id >= i)
				warp_sum += n;
		}
		s_sums[lane_id] = warp_sum;
	}
	__syncthreads();

	if (warp_id > 0)
		value += s_sums[warp_id - 1];
	__syncthreads();
}

// // taken from https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// __device__ void scan_inblock_workefficient_sync(int &value, void* smem_ptr, int lane_id, int warp_id, int warps_count)
// {
// 	int* s_values = (int*)smem_ptr;
// 	int n = warps_count * warpSize;
// 	int offset = 1;
	
// 	s_values[threadIdx.x] = value;

// 	// up-sweep
// 	for	(int d = n >> 1; d > 0; d >>= 1)
// 	{
// 		__syncthreads();
// 		if (threadIdx.x < d)
// 		{
// 			int ai = offset*(2*threadIdx.x + 1) - 1;
// 			int bi = offset*(2*threadIdx.x + 2) - 1;
// 			s_values[bi] += s_values[ai];
// 		}
// 		offset *= 2;
// 	}
	
// 	// down-sweep
// 	if (threadIdx.x == 0)
// 		s_values[n - 1] = 0;
// 	for (int d = 1; d < n; d *= 2)
// 	{
// 		offset >>= 1;
// 		__syncthreads();
// 		if (threadIdx.x < d)
// 		{
// 			int ai = offset*(2*threadIdx.x + 1) - 1;
// 			int bi = offset*(2*threadIdx.x + 2) - 1;
// 			int t = s_values[ai];
// 			s_values[ai] = s_values[bi];
// 			s_values[bi] += t;
// 		}
// 	}
// 	__syncthreads();
// 	if (threadIdx.x == n - 1)
// 		value = s_values[threadIdx.x] + s_values[1];
// 	else
// 		value = s_values[threadIdx.x + 1];
// 	__syncthreads();
// }

// // taken from https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// __device__ void scan_inblock_workefficient_noconflict_sync(int &value, void* smem_ptr, int lane_id, int warp_id, int warps_count)
// {
// 	int* s_values = (int*)smem_ptr;

// 	int n = warps_count * warpSize;
// 	int bankOffset = CONFLICT_FREE_OFFSET(threadIdx.x);
// 	s_values[threadIdx.x + bankOffset] = value;
	
// 	int offset = 1;
// 	// up-sweep
// 	for	(int d = n >> 1; d > 0; d >>= 1)
// 	{
// 		__syncthreads();
// 		if (threadIdx.x < d)
// 		{
// 			int ai = offset*(2*threadIdx.x + 1) - 1;
// 			int bi = offset*(2*threadIdx.x + 2) - 1;
// 			ai += CONFLICT_FREE_OFFSET(ai);
// 			bi += CONFLICT_FREE_OFFSET(bi);
// 			s_values[bi] += s_values[ai];
// 		}
// 		offset *= 2;
// 	}
	
// 	// down-sweep
// 	if (threadIdx.x == 0)
// 		s_values[n - 1 + CONFLICT_FREE_OFFSET(n-1)] = 0;
// 	for (int d = 1; d < n; d *= 2)
// 	{
// 		offset >>= 1;
// 		__syncthreads();
// 		if (threadIdx.x < d)
// 		{
// 			int ai = offset*(2*threadIdx.x + 1) - 1;
// 			int bi = offset*(2*threadIdx.x + 2) - 1;
// 			ai += CONFLICT_FREE_OFFSET(ai);
// 			bi += CONFLICT_FREE_OFFSET(bi);
// 			int t = s_values[ai];
// 			s_values[ai] = s_values[bi];
// 			s_values[bi] += t;
// 		}
// 	}
// 	__syncthreads();
// 	if (threadIdx.x == n - 1)
// 		value = s_values[threadIdx.x + bankOffset] + s_values[1 + CONFLICT_FREE_OFFSET(1)];
// 	else
// 		value = s_values[threadIdx.x + 1 + CONFLICT_FREE_OFFSET(threadIdx.x + 1)];
// 	__syncthreads();
// }

// // ==== UNROLLED ====

// // taken from https://github.com/NVIDIA/cuda-samples/blob/master/Samples/shfl_scan/shfl_scan.cu
// __device__ void inclusive_scan_inblock_shfl_unrolled_sync(int &value, void* smem_ptr, int lane_id, int warp_id, int warps_count) // niepotrzebny argument
// {
// 	int* s_sums = (int*)smem_ptr;

// 	// in-warp scan
//     #pragma unroll
// 	for (int i = 1; i <= WARPSIZE; i *= 2)
// 	{
// 		int n = __shfl_up_sync(FULL_MASK, value, i);	// add width as argument???

// 		if (lane_id >= i)
// 			value += n;
// 	}

// 	// inter-warp scan
// 	if (lane_id == WARPSIZE - 1)
// 		s_sums[warp_id] = value;
// 	__syncthreads();

// 	// the same shfl scan operation, but performed on warp sums
// 	// this can be safely done by a single warp, since there is maximum of 32 warps in a block
// 	if (warp_id == 0 && lane_id < WARPSCOUNT)
// 	{
// 		int warp_sum = s_sums[lane_id];

// 		int mask = (1 << WARPSCOUNT) - 1;
//         #pragma unroll
// 		for (int i = 1; i <= WARPSCOUNT; i *= 2)
// 		{
// 			int n = __shfl_up_sync(mask, warp_sum, i);
// 			if (lane_id >= i)
// 				warp_sum += n;
// 		}
// 		s_sums[lane_id] = warp_sum;
// 	}
// 	__syncthreads();

// 	if (warp_id > 0)
// 		value += s_sums[warp_id - 1];
// 	__syncthreads();
// }

// // taken from https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// __device__ void scan_inblock_workefficient_unrolled_sync(int &value, void* smem_ptr, int lane_id, int warp_id, int warps_count)
// {
// 	int* s_values = (int*)smem_ptr;
// 	// int n = warps_count * warpSize;
// 	int offset = 1;
	
// 	s_values[threadIdx.x] = value;

// 	// up-sweep
//     #pragma unroll
// 	for	(int d = BLOCKSIZE >> 1; d > 0; d >>= 1)
// 	{
// 		__syncthreads();
// 		if (threadIdx.x < d)
// 		{
// 			int ai = offset*(2*threadIdx.x + 1) - 1;
// 			int bi = offset*(2*threadIdx.x + 2) - 1;
// 			s_values[bi] += s_values[ai];
// 		}
// 		offset *= 2;
// 	}
	
// 	// down-sweep
// 	if (threadIdx.x == 0)
// 		s_values[BLOCKSIZE - 1] = 0;
//     #pragma unroll
// 	for (int d = 1; d < BLOCKSIZE; d *= 2)
// 	{
// 		offset >>= 1;
// 		__syncthreads();
// 		if (threadIdx.x < d)
// 		{
// 			int ai = offset*(2*threadIdx.x + 1) - 1;
// 			int bi = offset*(2*threadIdx.x + 2) - 1;
// 			int t = s_values[ai];
// 			s_values[ai] = s_values[bi];
// 			s_values[bi] += t;
// 		}
// 	}
// 	__syncthreads();
// 	if (threadIdx.x == BLOCKSIZE - 1)
// 		value = s_values[threadIdx.x] + s_values[1];
// 	else
// 		value = s_values[threadIdx.x + 1];
// 	__syncthreads();
// }

// // taken from https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// __device__ void scan_inblock_workefficient_noconflict_unrolled_sync(int &value, void* smem_ptr, int lane_id, int warp_id, int warps_count)
// {
// 	int* s_values = (int*)smem_ptr;

// 	int bankOffset = CONFLICT_FREE_OFFSET(threadIdx.x);
// 	s_values[threadIdx.x + bankOffset] = value;
	
// 	int offset = 1;
// 	// up-sweep
//     #pragma unroll
// 	for	(int d = BLOCKSIZE >> 1; d > 0; d >>= 1)
// 	{
// 		__syncthreads();
// 		if (threadIdx.x < d)
// 		{
// 			int ai = offset*(2*threadIdx.x + 1) - 1;
// 			int bi = offset*(2*threadIdx.x + 2) - 1;
// 			ai += CONFLICT_FREE_OFFSET(ai);
// 			bi += CONFLICT_FREE_OFFSET(bi);
// 			s_values[bi] += s_values[ai];
// 		}
// 		offset *= 2;
// 	}
	
// 	// down-sweep
// 	if (threadIdx.x == 0)
// 		s_values[BLOCKSIZE - 1 + CONFLICT_FREE_OFFSET(BLOCKSIZE-1)] = 0;
//     #pragma unroll
// 	for (int d = 1; d < BLOCKSIZE; d *= 2)
// 	{
// 		offset >>= 1;
// 		__syncthreads();
// 		if (threadIdx.x < d)
// 		{
// 			int ai = offset*(2*threadIdx.x + 1) - 1;
// 			int bi = offset*(2*threadIdx.x + 2) - 1;
// 			ai += CONFLICT_FREE_OFFSET(ai);
// 			bi += CONFLICT_FREE_OFFSET(bi);
// 			int t = s_values[ai];
// 			s_values[ai] = s_values[bi];
// 			s_values[bi] += t;
// 		}
// 	}
// 	__syncthreads();
// 	if (threadIdx.x == BLOCKSIZE - 1)
// 		value = s_values[threadIdx.x + bankOffset] + s_values[1 + CONFLICT_FREE_OFFSET(1)];
// 	else
// 		value = s_values[threadIdx.x + 1 + CONFLICT_FREE_OFFSET(threadIdx.x + 1)];
// 	__syncthreads();
// }
