#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

__global__ void hello_world_cuda();
int hello_world_cuda_host();