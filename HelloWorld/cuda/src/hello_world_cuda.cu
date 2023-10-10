#include <stdio.h>

__global__ void hello_world_cuda() {
    if (blockIdx.x==0 && threadIdx.x==0) printf("Hello world from GPU.\n");
    return;
}

int main() {
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    int computeCapabilityMajor = props.major;
    int computeCapabilityMinor = props.minor;
    int multiProcessorCount = props.multiProcessorCount;
    printf("Device ID: %d\nNumber of SMs: %d\nCompute Capability %d.%d\n", deviceId, multiProcessorCount, computeCapabilityMajor, computeCapabilityMinor);
    hello_world_cuda<<<16, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}