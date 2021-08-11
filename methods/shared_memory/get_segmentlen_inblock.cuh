#include "cuda_runtime.h"
#include "../../bit_functions.cuh"

#ifndef GET_SEGMENTLEN_FUN_ENUM
#define GET_SEGMENTLEN_FUN_ENUM
enum GET_SEGMENTLEN_FUN {
	GET_SEGMENTLEN_INBLOCK_AOS,
	GET_SEGMENTLEN_INBLOCK_SOA
};
#endif

__device__ int get_segmentlen_inblock_soa_sync(bool &is_begin, WORD_TYPE w_type, void* smem_ptr, int lane_id, int warp_id, int warps_count);
__device__ int get_segmentlen_inblock_aos_sync(bool &is_begin, WORD_TYPE w_type, void* smem_ptr, int lane_id, int warp_id, int warps_count);

__device__ int call_get_segmentlen_inblock(GET_SEGMENTLEN_FUN get_segmentlen_inblock, bool &is_begin, WORD_TYPE w_type, void* smem_ptr, int lane_id, int warp_id, int warps_count);
