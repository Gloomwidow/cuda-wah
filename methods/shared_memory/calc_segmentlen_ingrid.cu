// #include "smem_functions.cuh"


// __device__ void calc_segmentlen_ingrid_noncoalesced_sync(int &segment_len, int &index, bool &is_begin, WORD_TYPE w_type, void* smem_ptr, void* gmem_ptr, int lane_id, int warp_id, int warps_count, cg::grid_group grid)
// {
// 	segment* g_block_segments = (segment*)gmem_ptr;														// global memory allocation is just being reused
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

// 			g_block_segments[blockIdx.x].r_end_type = w_type;
// 			g_block_segments[blockIdx.x].r_end_len = segment_len;
// 		}
// 	}
// 	if (threadIdx.x == 0)						// first thread-beginning in block
// 	{
// 		g_block_segments[blockIdx.x].l_end_type = w_type;
// 		g_block_segments[blockIdx.x].l_end_len = segment_len;
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
// 			g_block_segments[blockIdx.x - 1].r_end_type == w_type)										// thread-beginning in the context of the grid...
// 		{
// 			is_begin = false;
// 			am_last_beginning_inblock = false;
// 			*s_decrement_index = true;
// 		}

// 		if (am_last_beginning_inblock)																	// ...if not, the last thread-beginning form prev. block should add sth to its `segment_len`
// 		{
// 			for (int i = blockIdx.x + 1; i < gridDim.x && g_block_segments[i].l_end_type == w_type; i++)
// 			{
// 				segment_len += g_block_segments[i].l_end_len;		// check types
// 				if (g_block_segments[i].l_end_len != blockDim.x)
// 					break;
// 			}
// 		}
// 	}
// 	__syncthreads();
// 	if (*s_decrement_index)																				// if the first thread is no longer thread-beginning,
// 		index -= 1;																			            // whole block needs to decrement index

// 	grid.sync();
// }

// __device__ void calc_segmentlen_ingrid_coalesced_sync(int &segment_len, int &index, bool &is_begin, WORD_TYPE w_type, void* smem_ptr, void* gmem_ptr, int lane_id, int warp_id, int warps_count, cg::grid_group grid)
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
// 	// return am_last_beginning_inblock;
// }

// __device__ void calc_segmentlen_ingrid_noncoalesced_sync(int* segment_len, int* index, bool* is_begin, WORD_TYPE w_type, void* smem_ptr, void* gmem_ptr, int lane_id, int warp_id, int warps_count, cg::grid_group grid)
// {
// 	segment* g_block_segments = (segment*)gmem_ptr;														// global memory allocation is just being reused
// 	unsigned* s_block_begins_masks = (unsigned*)smem_ptr;

// 	// ==== find the last thread-beginning in block ====
// 	unsigned warp_begins_mask = __ballot_sync(FULL_MASK, *is_begin);
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

// 			g_block_segments[blockIdx.x].r_end_type = w_type;
// 			g_block_segments[blockIdx.x].r_end_len = *segment_len;
// 		}
// 	}
// 	if (threadIdx.x == 0)						// first thread-beginning in block
// 	{
// 		g_block_segments[blockIdx.x].l_end_type = w_type;
// 		g_block_segments[blockIdx.x].l_end_len = *segment_len;
// 	}
// 	grid.sync();

// 	// ==== update segment_len and is_begin so that it's correct in grid boundaries ====
// 	bool* s_decrement_index = (bool*)smem_ptr;
// 	if (threadIdx.x == 0)
// 		*s_decrement_index = false;
// 	__syncthreads();
// 	if (*is_begin)
// 	{
// 		if (blockIdx.x > 0 && threadIdx.x == 0 && w_type != TAIL_WORD &&								// check if the first thread-beginning in block is really
// 			g_block_segments[blockIdx.x - 1].r_end_type == w_type)										// thread-beginning in the context of the grid...
// 		{
// 			*is_begin = false;
// 			am_last_beginning_inblock = false;
// 			*s_decrement_index = true;
// 		}

// 		if (am_last_beginning_inblock)																	// ...if not, the last thread-beginning form prev. block should add sth to its `segment_len`
// 		{
// 			for (int i = blockIdx.x + 1; i < gridDim.x && g_block_segments[i].l_end_type == w_type; i++)
// 			{
// 				*segment_len = (*segment_len) + g_block_segments[i].l_end_len;		// check types
// 				if (g_block_segments[i].l_end_len != blockDim.x)
// 					break;
// 			}
// 		}
// 	}
// 	__syncthreads();
// 	if (*s_decrement_index)																				// if the first thread is no longer thread-beginning,
// 		*index = (*index) - 1;																			// whole block needs to decrement index

// 	grid.sync();
// 	// return am_last_beginning_inblock;
// }

// __device__ void calc_segmentlen_ingrid_coalesced_sync(int* segment_len, int* index, bool* is_begin, WORD_TYPE w_type, void* smem_ptr, void* gmem_ptr, int lane_id, int warp_id, int warps_count, cg::grid_group grid)
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
// 	unsigned warp_begins_mask = __ballot_sync(FULL_MASK, *is_begin);
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
// 			g_block_segments.r_end_len[blockIdx.x] = *segment_len;
// 		}
// 	}
// 	if (threadIdx.x == 0)						// first thread-beginning in block
// 	{
// 		g_block_segments.l_end_type[blockIdx.x] = w_type;
// 		g_block_segments.l_end_len[blockIdx.x] = *segment_len;
// 	}
// 	grid.sync();

// 	// ==== update segment_len and is_begin so that it's correct in grid boundaries ====
// 	bool* s_decrement_index = (bool*)smem_ptr;
// 	if (threadIdx.x == 0)
// 		*s_decrement_index = false;
// 	__syncthreads();
// 	if (*is_begin)
// 	{
// 		if (blockIdx.x > 0 && threadIdx.x == 0 && w_type != TAIL_WORD &&								// check if the first thread-beginning in block is really
// 			g_block_segments.r_end_type[blockIdx.x] == w_type)  										// thread-beginning in the context of the grid...
// 		{
// 			*is_begin = false;
// 			am_last_beginning_inblock = false;
// 			*s_decrement_index = true;
// 		}

// 		if (am_last_beginning_inblock)																	// ...if not, the last thread-beginning form prev. block should add sth to its `segment_len`
// 		{
// 			for (int i = blockIdx.x + 1; i < gridDim.x && g_block_segments.l_end_type[i] == w_type; i++)
// 			{
// 				*segment_len = (*segment_len) + g_block_segments.l_end_len[i];
// 				if (g_block_segments.l_end_len[i] != blockDim.x)
// 					break;
// 			}
// 		}
// 	}
// 	__syncthreads();
// 	if (*s_decrement_index)																				// if the first thread is no longer thread-beginning,
// 		*index = (*index) - 1;																			// whole block needs to decrement index

// 	grid.sync();
// 	// return am_last_beginning_inblock;
// }
