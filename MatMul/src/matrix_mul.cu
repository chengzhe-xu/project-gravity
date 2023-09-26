#include "include/matrix_mul.h"

__global__ void matrix_mul_naive_kernel_32x32(float* matA, float* matB, float* matC, int M, int N, int K) {
    const unsigned int block_id = blockIdx.x;
    const unsigned int thread_id = threadIdx.x;
    const unsigned int block_row = block_id / 16;
    const unsigned int block_col = block_id % 16;
    const unsigned int thread_row = thread_id / 32;
    const unsigned int thread_col = thread_id % 32;

    int C_row = 32 * block_row + thread_row;
    int C_col = 32 * block_col + thread_col;

    // fma
    for (int i=0; i<K; ++i) {
        // C[C_row, C_col] += A[C_row, i] * B[i, C_col]
        matC[N * C_row + C_col] += matA[K * C_row + i] * matB[N * i + C_col];
    }
    __syncthreads();
    return;
}

void matrix_mul_naive_host(const float* matA, const float* matB, float* matC, int M, int N, int K) {
    // cudaMalloc device arrays
    float* device_matA = 0;
    float* device_matB = 0;
    float* device_matC = 0;
    cudaMalloc((void**)&device_matA, M * K * sizeof(float));
    cudaMalloc((void**)&device_matB, N * K * sizeof(float));
    cudaMalloc((void**)&device_matC, M * N * sizeof(float));
    if(device_matA == 0 || device_matB == 0 || device_matC == 0) {
        printf("couldn't allocate memory\n");
        return;
    }
    // cuda mem copy
    cudaMemcpy(device_matA, matA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_matB, matB, N * K * sizeof(float), cudaMemcpyHostToDevice);
    // kernel call
    // note that the size is 2048 * 512, we choose 32 * 32 kernels
    int block_size = 32 * 32;
    int grid_size = (M * N) / block_size;
    matrix_mul_naive_kernel_32x32<<<grid_size, block_size>>>(device_matA, device_matB, device_matC, M, N, K);
    cudaMemcpy(matC, device_matC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_matA);
    cudaFree(device_matB);
    cudaFree(device_matC);
    return;
}
