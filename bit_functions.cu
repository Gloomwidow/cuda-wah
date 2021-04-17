//#include "bit_functions.cuh"
//#ifndef UINT
//#define UINT unsigned int
//#endif // !UINT
//#ifndef ULONG
//#define ULONG unsigned long long
//#endif // !ULONG
//
////gets i-th bit in int
//__device__ UINT get_bit(UINT src, int i)
//{
//	return (1 & (src >> (31 - (i))));
//}
////sets u-th bit to 1
//__device__ UINT fill_bit(UINT src, int i)
//{
//	src |= 1UL << (31 - i);
//	return src;
//}
////sets u-th bit to 0
//__device__ UINT clear_bit(UINT src, int i)
//{
//	src &= ~(1UL << (31 - i));
//	return src;
//}
////Compresses n blocks of 'bit'
////1st bit - is compressed flag
////2nd bit - what bit is in series (all 0 or all 1)
////3rd to end - block count
//__device__ UINT get_compressed(UINT n, int bit)
//{
//	UINT rs = n;
//	rs = fill_bit(rs, 0);
//	if (bit) rs = fill_bit(rs, 1);
//	return rs;
//}
//
////checks if all bits in src are zero
//__device__ bool is_zeros(UINT src)
//{
//	return src == 0;
//}
////checks if all bits in src are ones
//__device__ bool is_ones(UINT src)
//{
//	src = fill_bit(src, 0);
//	return (~src) == 0;
//}
//
//__device__ UINT reverse(UINT src)
//{
//	UINT NO_OF_BITS = 32;
//	UINT reverse_num = 0, i, temp;
//
//	for (i = 0; i < NO_OF_BITS; i++)
//	{
//		temp = (src & (1 << i));
//		if (temp) reverse_num |= (1 << ((NO_OF_BITS - 1) - i));
//	}
//	return reverse_num;
//}
