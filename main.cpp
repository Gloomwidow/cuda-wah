#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ZerosGenerator.h"
#include <cstdio>
#include <climits>
#include <bitset>
#include <iostream>

extern void CudaHello();

//test function for bitwise operations
void bits(int a)
{
    std::bitset<32> x(a);
    std::cout << x << '\n';
}

int main() {
    printf("Hello host!\n");
    CudaHello(); 
    IntDataGenerator* gen = new ZerosGenerator();
    UINT* tab = gen->GetHostData(15);
    for (int i = 0; i < 15; i++)
    {
        printf("%u ", tab[i]);
    }
    printf("\n");
    bits(1 << 31);
    bits((1 << 32));
    bits(~(1 << 32));
    printf("%d\n",(1 & (2345 >> (6 - 1))));
    return 0;
}