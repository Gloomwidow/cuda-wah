#include "device_launch_parameters.h"
#include "defines.h"

//gets i-th bit in int
__device__ UINT get_bit(UINT src, int i);

//sets u-th bit to 1
__host__ __device__ UINT fill_bit(UINT src, int i);

//sets u-th bit to 0
__host__ __device__ UINT clear_bit(UINT src, int i);

//checks if all bits in src are zero
__device__ bool is_zeros(UINT src);

//checks if all bits in src are ones
__device__ bool is_ones(UINT src);

//Compresses n blocks of 'bit'
//1st bit - is compressed flag
//2nd bit - what bit is in series (all 0 or all 1)
//3rd to end - block count
__device__ UINT get_compressed(UINT n, int bit);

__device__ UINT reverse(UINT src);

//returns amount of sequences compressed in src block
__host__ __device__ UINT compressed_count(UINT src);
