#include <cstdio>
extern void cudaHello();
int main() {
    printf("Hello host!\n");
    cudaHello(); 
    return 0;
}