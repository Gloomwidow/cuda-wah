//#include "cuda.h"
#include "device_launch_parameters.h"

#ifndef UINT
#define UINT unsigned int
#endif // !UINT
#ifndef ULONG
#define ULONG unsigned long long
#endif // !ULONG


__global__ void SharedMemKernel(UINT* input)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int gulp = input[idx];

	//bool is_zero = 
}

void SharedMemWAH(UINT* input, size_t size)
{

}
