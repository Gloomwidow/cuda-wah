#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #include <cooperative_groups.h>
#include "../../bit_functions.cuh"
// namespace cg = cooperative_groups;

typedef int (*get_segmentlen_inblock_fun)(bool &is_begin, WORD_TYPE w_type, void* smem_ptr, int lane_id, int warp_id, int warps_count);
// __device__ int get_segmentlen_inblock_soa_sync(bool &is_begin, WORD_TYPE w_type, void* smem_ptr, int lane_id, int warp_id, int warps_count);
__device__ int get_segmentlen_inblock_aos_sync(bool &is_begin, WORD_TYPE w_type, void* smem_ptr, int lane_id, int warp_id, int warps_count);

typedef void (*inclusive_scan_inblock_fun)(int& value, void* smem_ptr, int lane_id, int warp_id, int warps_count);
__device__ void inclusive_scan_inblock_shfl_sync(int &value, void* smem_ptr, int lane_id, int warp_id, int warps_count);
// __device__ void scan_inblock_workefficient_sync(int &value, void* smem_ptr, int lane_id, int warp_id, int warps_count);
// __device__ void scan_inblock_workefficient_noconflict_sync(int &value, void* smem_ptr, int lane_id, int warp_id, int warps_count);
// __device__ void inclusive_scan_inblock_shfl_unrolled_sync(int &value, void* smem_ptr, int lane_id, int warp_id, int warps_count); // niepotrzebny argument
// __device__ void scan_inblock_workefficient_unrolled_sync(int &value, void* smem_ptr, int lane_id, int warp_id, int warps_count);
// __device__ void scan_inblock_workefficient_noconflict_unrolled_sync(int &value, void* smem_ptr, int lane_id, int warp_id, int warps_count);

// typedef void (*calc_segmentlen_ingrid_fun)(int& segment_len, int& index, bool& is_begin, WORD_TYPE w_type, void* smem_ptr, void* gmem_ptr, int lane_id, int warp_id, int warps_count, cg::grid_group grid);
// __device__ void calc_segmentlen_ingrid_noncoalesced_sync(int &segment_len, int &index, bool &is_begin, WORD_TYPE w_type, void* smem_ptr, void* gmem_ptr, int lane_id, int warp_id, int warps_count, cg::grid_group grid);
// __device__ void calc_segmentlen_ingrid_coalesced_sync(int &segment_len, int &index, bool &is_begin, WORD_TYPE w_type, void* smem_ptr, void* gmem_ptr, int lane_id, int warp_id, int warps_count, cg::grid_group grid);
