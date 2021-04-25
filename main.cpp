#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ZerosGenerator.h"
#include "defines.h"
#include "methods.h"
#include "wah_test.h"
#include <cstdio>
#include <climits>
#include <bitset>
#include <iostream>


int main() 
{
    if (UNIT_TESTING)
    {
        UnitTests(&BallotSyncWAH);
    }
    else
    {
        UINT size = 32 * 20000000;
        printf("Generating tests...\n");
        UINT* data = new UINT[size];
        for (int i = 0; i < size; i++)
        {
            int roll = rand() % 3;
            if (roll == 0) data[i] = 0;
            else if (roll == 1) data[i] = 0x7FFFFFFF;
            else data[i] = 32;
        }
        UINT* d_data;
        cudaMalloc((UINT**)&d_data, sizeof(UINT) * size);
        cudaMemcpy(d_data, data, sizeof(UINT) * size, cudaMemcpyHostToDevice);
        delete[] data;
        Benchmark(&BallotSyncWAH, size, d_data, 10, "1. __ballot_sync() + 2x thrust::remove_if", true);
        cudaFree(d_data);      
    }
    return 0;
}