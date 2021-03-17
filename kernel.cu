#include <cstdio>
__global__ void cuda_hello(){
    printf("Hello device!\n");
}

void cudaHello()
{
    printf("Hello extern!\n");
    cuda_hello<<<1,1>>>(); 
}


