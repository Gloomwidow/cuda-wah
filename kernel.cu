#include <cstdio>
#ifndef UINT
#define UINT unsigned int
#endif // !UINT
#ifndef ULONG
#define ULONG unsigned long long
#endif // !ULONG

__global__ void cuda_hello(){
    //printf("%d\n",blockIdx.x);
}

//gets i-th bit in int
__device__ UINT get_bit(UINT src,int i)
{
    return (1 & (src >> (i - 1)));
}
//sets u-th bit to 1
__device__ UINT fill_bit(UINT src, int i)
{
    src |= 1UL << i;
    return src;
}
//sets u-th bit to 0
__device__ UINT clear_bit(UINT src, int i)
{
    src &= ~(1UL << i);
    return src;
}
//Compresses n blocks of 'bit'
//1st bit - is compressed flag
//2nd bit - what bit is in series (all 0 or all 1)
//3rd to end - block count
__device__ UINT get_compressed(UINT n, int bit)
{
    UINT rs = n;
    rs = fill_bit(rs, 31);
    if(bit) rs = fill_bit(rs, 30);
    return rs;
}


void CudaHello()
{
    printf("Hello extern!\n");
    cuda_hello<<<4,4>>>(); 
}


