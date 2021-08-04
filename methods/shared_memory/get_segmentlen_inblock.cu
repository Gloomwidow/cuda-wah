#include "cuda_runtime.h"
#include "../../bit_functions.cuh"

typedef int (*get_segmentlen_inblock_fun)(bool &is_begin, WORD_TYPE w_type, void* smem_ptr, int lane_id, int warp_id, int warps_count);

__device__ int get_segmentlen_inblock_soa_sync(bool &is_begin, WORD_TYPE w_type, void* smem_ptr, int lane_id, int warp_id, int warps_count)
{
	// segment* segments = (segment*)smem_ptr;
    segment_soa segments;
    segments.l_end_type = (WORD_TYPE*)smem_ptr;
    segments.l_end_len = (int*)(&segments.l_end_type[warps_count]);
    segments.r_end_type = (WORD_TYPE*)(&segments.l_end_len[warps_count]);
    segments.r_end_len = (int*)(&segments.r_end_type[warps_count]);

	bool am_last_beginning_inwarp = false;
	unsigned warp_begins_mask = __ballot_sync(FULL_MASK, is_begin);
	int segment_len = 0;
	if (is_begin)
	{																									// find ID of the next thread-beginning and thus the length of the segment
		segment_len = (lane_id == warpSize - 1) ? 0 : __ffs(warp_begins_mask >> (lane_id + 1));			// note: bit shift "(int) >> 32" is not defined
																										// note: __ffs(0) = 0
		if (segment_len == 0)	// the last thread-beginning in warp
		{
			am_last_beginning_inwarp = true;
			segment_len = (warp_id < warps_count - 1) ? (warpSize - lane_id) : (warps_count * warpSize - threadIdx.x);
			// considers case of the last thread-beginning in the last warp in block
			// when inputSize is not divisible by 32
			segments.r_end_type[warp_id] = w_type;
			segments.r_end_len[warp_id] = segment_len;
		}
		if (lane_id == 0)		// the first thread-beginning in warp
		{
			segments.l_end_type[warp_id] = w_type;
			segments.l_end_len[warp_id] = segment_len;
		}
	}
	__syncthreads();

	if (is_begin)
	{
		if (warp_id > 0 && lane_id == 0 && w_type != TAIL_WORD &&										// check if the first thread-beginning in warp is really
			(segments.r_end_type[warp_id - 1] == w_type))												// thread-beginning in the context of the block...
		{
			is_begin = false;
			am_last_beginning_inwarp = false;
		}

		if (am_last_beginning_inwarp)																	// ...if not, the last thread-beginning form prev. warp should add sth to its `segment_len`
		{
			for (int i = warp_id + 1; i < warps_count && segments.l_end_type[i] == w_type; i++)
			{
				segment_len += segments.l_end_len[i];		// check types
				if (segments.l_end_len[i] != warpSize)
					break;
			}
		}
	}
	__syncthreads();

	return segment_len;
}

__device__ int get_segmentlen_inblock_aos_sync(bool &is_begin, WORD_TYPE w_type, void* smem_ptr, int lane_id, int warp_id, int warps_count)
{
	segment* segments = (segment*)smem_ptr;

	bool am_last_beginning_inwarp = false;
	unsigned warp_begins_mask = __ballot_sync(FULL_MASK, is_begin);
	int segment_len = 0;
	if (is_begin)
	{																									// find ID of the next thread-beginning and thus the length of the segment
		segment_len = (lane_id == warpSize - 1) ? 0 : __ffs(warp_begins_mask >> (lane_id + 1));			// note: bit shift "(int) >> 32" is not defined
																										// note: __ffs(0) = 0
		if (segment_len == 0)	// the last thread-beginning in warp
		{
			am_last_beginning_inwarp = true;
			segment_len = (warp_id < warps_count - 1) ? (warpSize - lane_id) : (warps_count * warpSize - threadIdx.x);
			// considers case of the last thread-beginning in the last warp in block
			// when inputSize is not divisible by 32
			segments[warp_id].r_end_type = w_type;
			segments[warp_id].r_end_len = segment_len;
		}
		if (lane_id == 0)		// the first thread-beginning in warp
		{
			segments[warp_id].l_end_type = w_type;
			segments[warp_id].l_end_len = segment_len;
		}
	}
	__syncthreads();

	if (is_begin)
	{
		if (warp_id > 0 && lane_id == 0 && w_type != TAIL_WORD &&										// check if the first thread-beginning in warp is really
			(segments[warp_id - 1].r_end_type == w_type))												// thread-beginning in the context of the block...
		{
			is_begin = false;
			am_last_beginning_inwarp = false;
		}

		if (am_last_beginning_inwarp)																	// ...if not, the last thread-beginning form prev. warp should add sth to its `segment_len`
		{
			for (int i = warp_id + 1; i < warps_count && segments[i].l_end_type == w_type; i++)
			{
				segment_len += segments[i].l_end_len;		// check types
				if (segments[i].l_end_len != warpSize)
					break;
			}
		}
	}
	__syncthreads();

	return segment_len;
}
