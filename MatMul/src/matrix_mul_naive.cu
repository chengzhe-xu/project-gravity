#include "matrix_mul.h"
#include "cuda_utils.h"

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

__global__ void matrix_mul_naive_kernel_32x32(__half* matAT, __half* matB, __half* matC, int M, int N, int K) {
    const unsigned int block_id = blockIdx.x;
    const unsigned int thread_id = threadIdx.x;
    const unsigned int block_row = block_id / 16;
    const unsigned int block_col = block_id % 16;
    const unsigned int thread_row = thread_id / 32;
    const unsigned int thread_col = thread_id % 32;

    int C_row = 32 * block_row + thread_row;
    int C_col = 32 * block_col + thread_col;

    // fma
    __half tmp_C = matC[N * C_row + C_col];
    #pragma unroll
    for (int i=0; i<K; ++i) {
        // C[C_row, C_col] += AT[i, C_row] * B[i, C_col]
        tmp_C = __hfma(matAT[M * i + C_row], matB[N * i + C_col], tmp_C); // __half
        // tmp_C += matAT[M * i + C_row] * matB[N * i + C_col]; // float
        // initially it was:
        // matC[N * C_row + C_col] += matA[K * C_row + i] * matB[N * i + C_col];
    }
    matC[N * C_row + C_col] = tmp_C;
    __syncthreads();
    return;
}

matrix_template matrix_mul_naive_host(const matrix_template& matAT, const matrix_template& matB, matrix_template& matC, int M, int N, int K) {
    event_pair timer;
    // cudaMalloc device arrays
    float* device_matAT = 0;
    float* device_matB = 0;
    float* device_matC = 0;
    cudaMalloc((void**)&device_matAT, M * K * sizeof(float));
    cudaMalloc((void**)&device_matB, N * K * sizeof(float));
    cudaMalloc((void**)&device_matC, M * N * sizeof(float));
    if(device_matAT == 0 || device_matB == 0 || device_matC == 0) {
        printf("couldn't allocate memory\n");
        return matC;
    }
    // __half_copy
    __half* device_matAT_h = 0;
    __half* device_matB_h = 0;
    __half* device_matC_h = 0;
    cudaMalloc((void**)&device_matAT_h, M * K * sizeof(__half));
    cudaMalloc((void**)&device_matB_h, N * K * sizeof(__half));
    cudaMalloc((void**)&device_matC_h, M * N * sizeof(__half));
    if(device_matAT_h == 0 || device_matB_h == 0 || device_matC_h == 0) {
        printf("couldn't allocate memory\n");
        return matC;
    }
    // cuda mem copy
    cudaMemcpy(device_matAT, matAT.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_matB, matB.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_matC, matC.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    cast_kernel_float2half<<<128, 256>>>(device_matAT_h, device_matAT, M * K);
    cast_kernel_float2half<<<128, 256>>>(device_matB_h, device_matB, N * K);
    cast_kernel_float2half<<<128, 256>>>(device_matC_h, device_matC, M * N);

    // kernel call
    // note that the size is 2048 * 512, we choose 32 * 32 kernels
    int block_size = 32 * 32;
    int grid_size = (M * N) / block_size;
    start_timer(&timer);
    matrix_mul_naive_kernel_32x32<<<grid_size, block_size>>>(device_matAT_h, device_matB_h, device_matC_h, M, N, K);
    float kernel_time_ms = stop_timer(&timer);
    cast_kernel_half2float<<<128, 256>>>(device_matC, device_matC_h, M * N);
    cudaMemcpy(matC.data(), device_matC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_matAT);
    cudaFree(device_matB);
    cudaFree(device_matC);
    cudaFree(device_matAT_h);
    cudaFree(device_matB_h);
    cudaFree(device_matC_h);
    printf("cuda kernel <matrix_mul_naive_kernel_32x32> runtime %f ms.\n", kernel_time_ms);
    return matC;
}
