#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ZerosGenerator.h"
#include <cstdio>

extern void CudaHello();

int main() {
    printf("Hello host!\n");
    CudaHello(); 
    IntDataGenerator* gen = new ZerosGenerator();
    UInt* tab = gen->GetDeviceData(15);
    for (int i = 0; i < 15; i++)
    {
        printf("%u ", tab[i]);
    }
    printf("\n");
    
    return 0;
}