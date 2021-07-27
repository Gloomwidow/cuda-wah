#include "device_launch_parameters.h"
#include "defines.h"


enum WORD_TYPE {
	EMPTY_WORD = 0,
	FULL_WORD = 1,
	TAIL_WORD = 2
};

__host__ __device__ WORD_TYPE get_word_type(UINT gulp);

//gets i-th bit in int
__host__ __device__ UINT get_bit(UINT src, int i);

//sets u-th bit to 1
__host__ __device__ UINT fill_bit(UINT src, int i);

//sets u-th bit to 0
__host__ __device__ UINT clear_bit(UINT src, int i);

//checks if all bits in src are zero
__host__ __device__ bool is_zeros(UINT src);

//checks if all bits in src are ones
__host__ __device__ bool is_ones(UINT src);

//Compresses n blocks of 'bit'
//1st bit - is compressed flag
//2nd bit - what bit is in series (all 0 or all 1)
//3rd to end - block count
__host__ __device__ UINT get_compressed(UINT n, int bit);

__host__ __device__ UINT reverse(UINT src);

//returns amount of sequences compressed in src block
__host__ __device__ UINT compressed_count(UINT src);

void printBits(size_t const size, void const * const ptr);

__global__ void ballot_warp_merge(int input_size, UINT* input, UINT* output);

struct wah_zero
{
	__host__ __device__
		bool operator()(const int x)
	{
		return x == 0;
	}
};