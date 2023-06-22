#include <stdio.h>

__global__ void hello_world_cuda() {
    if (blockIdx.x==0 && threadIdx.x==0) printf("Hello world from GPU.\n");
    return;
}

int main() {
    hello_world_cuda<<<16, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}