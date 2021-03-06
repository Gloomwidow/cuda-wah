#include "bit_functions.cuh"
#include <cstdio>


__host__ __device__ WORD_TYPE get_word_type(UINT gulp)
{
	if (is_zeros(gulp))
		return EMPTY_WORD;
	if (is_ones(gulp))
		return FULL_WORD;
	return TAIL_WORD;
}

__host__ __device__ UINT get_bit(UINT src, int i)
{
	return (1 & (src >> (31 - (i))));
}

__host__ __device__ UINT fill_bit(UINT src, int i)
{
	src |= 1UL << (31 - i);
	return src;
}

__host__ __device__ UINT clear_bit(UINT src, int i)
{
	src &= ~(1UL << (31 - i));
	return src;
}


__host__ __device__ bool is_zeros(UINT src)
{
	return src == 0;
}

__host__ __device__ bool is_ones(UINT src)
{
	src = fill_bit(src, 0);
	return (~src) == 0;
}

__host__ __device__ UINT get_compressed(UINT n, int bit)
{
	UINT rs = n;
	rs = fill_bit(rs, 0);
	if (bit) rs = fill_bit(rs, 1);
	return rs;
}

__host__ __device__ UINT reverse(UINT src)
{
	UINT NO_OF_BITS = 32;
	UINT reverse_num = 0, i, temp;

	for (i = 0; i < NO_OF_BITS; i++)
	{
		temp = (src & (1 << i));
		if (temp) reverse_num |= (1 << ((NO_OF_BITS - 1) - i));
	}
	return reverse_num;
}

//returns amount of sequences compressed in src block
__host__ __device__ UINT compressed_count(UINT src)
{
	src = clear_bit(src, 0);
	src = clear_bit(src, 1);
	return src;
}

void printBits(size_t const size, void const * const ptr)
{
	unsigned char *b = (unsigned char*)ptr;
	unsigned char byte;
	int i, j;

	for (i = size - 1; i >= 0; i--) {
		for (j = 7; j >= 0; j--) {
			byte = (b[i] >> j) & 1;
			printf("%u", byte);
		}
	}
	puts("");
}