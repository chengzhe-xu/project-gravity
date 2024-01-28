#include "mat_trans.cuh"

__global__ void cast_kernel_float2half(__half* arr_h, float* arr, const int arr_size) {
    const unsigned int stride = gridDim.x * blockDim.x;
    const unsigned int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll
    for (int i=start_idx; i<arr_size; i+=stride) {
        arr_h[i] = __float2half(arr[i]);
    }
    __syncthreads();
    return;
}

__global__ void cast_kernel_half2float(float* arr, __half* arr_h, const int arr_size) {
    const unsigned int stride = gridDim.x * blockDim.x;
    const unsigned int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll
    for (int i=start_idx; i<arr_size; i+=stride) {
        arr[i] = __half2float(arr_h[i]);
    }
    __syncthreads();
    return;
}
