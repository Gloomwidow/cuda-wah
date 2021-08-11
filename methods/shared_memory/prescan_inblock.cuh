#include "cuda_runtime.h"

#ifndef PRESCAN_INBLOCK_ENUM
#define PRESCAN_INBLOCK_ENUM
enum PRESCAN_INBLOCK_FUN {
	PRESCAN_SHFL,
	PRESCAN_WORKEFFICIENT,
	PRESCAN_WORKEFFICIENT_CONFLICTFREE
	#ifdef UNROLLED_PRESCANS
	,PRESCAN_SHFL_UNROLLED,
	PRESCAN_WORKEFFICIENT_UNROLLED,
	PRESCAN_WORKEFFICIENT_CONFLICTFREE_UNROLLED
	#endif
};
#endif

__device__ void prescan_inblock_shfl_sync(int &value, void* smem_ptr, int lane_id, int warp_id, int warps_count);
__device__ void prescan_inblock_workefficient_sync(int &value, void* smem_ptr, int lane_id, int warp_id, int warps_count);
__device__ void prescan_inblock_workefficient_conflictfree_sync(int &value, void* smem_ptr, int lane_id, int warp_id, int warps_count);
#ifdef UNROLLED_PRESCANS
__device__ void prescan_inblock_shfl_unrolled_sync(int &value, void* smem_ptr, int lane_id, int warp_id, int warps_count);
__device__ void prescan_inblock_workefficient_unrolled_sync(int &value, void* smem_ptr, int lane_id, int warp_id, int warps_count);
__device__ void prescan_inblock_workefficient_conflictfree_unrolled_sync(int &value, void* smem_ptr, int lane_id, int warp_id, int warps_count);
#endif

__device__ void call_prescan_inblock(PRESCAN_INBLOCK_FUN prescan_inblock, int &value, void* smem_ptr, int lane_id, int warp_id, int warps_count);
