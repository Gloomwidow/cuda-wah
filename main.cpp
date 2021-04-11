#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ZerosGenerator.h"
#include <cstdio>
#include <climits>
#include <bitset>
#include <iostream>

extern void CudaHello();
extern void BallotSyncWAH(UINT* input);

//test function for bitwise operations
void bits(int a)
{
    std::bitset<32> x(a);
    std::cout << x << '\n';
}

int main() {
    printf("Hello host!\n");
    BallotSyncWAH(nullptr);
    return 0;
}