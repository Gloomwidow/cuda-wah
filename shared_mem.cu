//#include "cuda.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "bit_functions.cuh"

__global__ void SharedMemKernel(UINT* input)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	UINT gulp = input[idx];

	bool is_zero = is_zeros(gulp);
	//bool is_one = is_ones(gulp);
	//bool is_zero = (gulp == 0);
	//bool is_one = is_ones(gulp);

	//__shuffle_up
}

void SharedMemWAH(UINT* input, size_t size)
{
	UINT* test = new UINT[size];
	UINT* output = new UINT[size];

	srand(time(NULL));
	for (int i = 0; i < size; i++)
	{
		int roll = rand() % 3;
		if (roll == 0)
		{
			test[i] = 0x7FFFFFFF; //all ones
			printf("1");
		}
		if (roll == 1)
		{
			test[i] = 256; // not valid for compression
			printf("x");
		}
		if (roll == 2)
		{
			test[i] = 0x00000000; // all zeros
			printf("0");
		}
	}
}
