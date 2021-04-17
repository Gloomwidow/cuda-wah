#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ZerosGenerator.h"
#include <cstdio>
#include <climits>
#include <bitset>
#include <iostream>
#include "defines.h"

extern void CudaHello();
extern void BallotSyncWAH(UINT* input);
extern void SharedMemWAH(UINT* input, size_t size);

//test function for bitwise operations
void bits(int a)
{
    std::bitset<32> x(a);
    std::cout << x << '\n';
}

int main() {
    SharedMemWAH(nullptr, 2);
	//BallotSyncWAH(nullptr);
    return 0;
}
