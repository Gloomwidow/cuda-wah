#include "device_launch_parameters.h"
#include "defines.h"

__device__ UINT get_bit(UINT src, int i);

__device__ UINT fill_bit(UINT src, int i);

__device__ UINT clear_bit(UINT src, int i);

__device__ bool is_zeros(UINT src);

__device__ bool is_ones(UINT src);

__device__ UINT get_compressed(UINT n, int bit);

__device__ UINT reverse(UINT src);
