#pragma once
#include "methods/shared_memory/smem_functions.cuh"
// #include "cuda_runtime.h"
// #include "cooperative_groups.h"
#ifndef WAH_FUN_TYPEDEF
#define WAH_FUN_TYPEDEF
typedef UINT* (*WAH_fun)(int size, UINT* input);
#endif // WAH_FUN_TYPEDEF
// namespace cg = cooperative_groups;

void WarmUp();
UINT* RemoveIfWAH(int data_size, UINT* input, int threads_per_block);
UINT* AtomicAddWAH(int data_size, UINT* input, int threads_per_block);

// typedef void (*calc_segmentlen_ingrid_fun)(int& segment_len, int& index, bool& is_begin, WORD_TYPE w_type, void* smem_ptr, void* gmem_ptr, int lane_id, int warp_id, int warps_count, cg::grid_group grid);
// template<get_segmentlen_inblock_fun get_segmentlen_inblock,
// 		inclusive_scan_inblock_fun inclusive_scan_inblock,
// 		calc_segmentlen_ingrid_fun calc_segmentlen_ingrid
// 		>
// UINT* SharedMemWAH(int data_size, UINT* input, int threads_per_block);

//optimized method variants
UINT* OptimizedRemoveIfWAH(int data_size, UINT* input, int threads_per_block);
UINT* OptimizedAtomicAddWAH(int data_size, UINT* input, int threads_per_block);

WAH_fun* get_wahs(int* count);
void run();
