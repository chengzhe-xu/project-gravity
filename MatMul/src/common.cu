#include "matrix_mul.cuh"

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

__device__ __forceinline__ void ldg128(const __half2* addr, __half2 &reg0, __half2 &reg1, __half2 &reg2, __half2 &reg3){
    asm volatile(
        "ld.global.nc.v4.b32 {%1, %2, %3, %4}, [%0];\n"
        : "=r"(__HALF2_TO_UI(reg0)),
          "=r"(__HALF2_TO_UI(reg1)),
          "=r"(__HALF2_TO_UI(reg2)),
          "=r"(__HALF2_TO_UI(reg3))
        : "l"(addr)
    );
}

__device__ __forceinline__ void stg128(__half2* addr, __half2 &reg0, __half2 &reg1, __half2 &reg2, __half2 &reg3) {
    asm volatile(
        "st.global.v4.b32 [%0], {%1, %2, %3, %4};\n"
        :
        : "l"(addr),
          "r"(__HALF2_TO_UI(reg0)),
          "r"(__HALF2_TO_UI(reg1)),
          "r"(__HALF2_TO_UI(reg2)),
          "r"(__HALF2_TO_UI(reg3))
    );
}

__device__ __forceinline__ void half2matmulacc(__half2 acc[8][4], __half2 pA[8], __half2 pB[8]) {
    // TODO
    return
}