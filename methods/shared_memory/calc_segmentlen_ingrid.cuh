#include "cuda_runtime.h"
#include "cooperative_groups.h"
#include "../../bit_functions.cuh"
namespace cg = cooperative_groups;

#ifndef CALC_SEGMENTLEN_INGRID_FUN_ENUM
#define CALC_SEGMENTLEN_INGRID_FUN_ENUM
enum CALC_SEGMENTLEN_INGRID_FUN {
	CALC_SEGMENTLEN_INGRID_NONCOALESCED,
	CALC_SEGMENTLEN_INGRID_COALESCED
};
#endif

__device__ void calc_segmentlen_ingrid_noncoalesced_sync(int &segment_len, int &index, bool &is_begin, WORD_TYPE w_type, void* smem_ptr, void* gmem_ptr, int lane_id, int warp_id, int warps_count, cg::grid_group grid);
__device__ void calc_segmentlen_ingrid_coalesced_sync(int &segment_len, int &index, bool &is_begin, WORD_TYPE w_type, void* smem_ptr, void* gmem_ptr, int lane_id, int warp_id, int warps_count, cg::grid_group grid);

__device__ void call_calc_segmentlen_ingrid(CALC_SEGMENTLEN_INGRID_FUN calc_segmentlen_ingrid, int &segment_len, int &index, bool &is_begin, WORD_TYPE w_type, void* smem_ptr, void* gmem_ptr, int lane_id, int warp_id, int warps_count, cg::grid_group grid);
