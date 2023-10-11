#include "matrix_mul.cuh"
#include "cuda_utils.h"

__global__ void matrix_mul_smit_kernel_32x32(__half* matA, __half* matBT, __half* matC, int M, int N, int K) {
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
    for (int i=0; i<K; ++i) {
        // C[C_row, C_col] += A[C_row, i] * BT[C_col, i]
        tmp_C = __hfma(matA[C_row * K+ i], matBT[C_col * K + i], tmp_C);
        // TODO: initially it was:
        // matC[N * C_row + C_col] += matA[K * C_row + i] * matB[N * i + C_col];
        // why the current one is better (2.28x faster)?
    }
    matC[N * C_row + C_col] = tmp_C;
    __syncthreads();
    return;
}

matrix_template matrix_mul_smit_host(const matrix_template& matA, const matrix_template& matBT, matrix_template& matC, int M, int N, int K) {
    event_pair timer;
    // cudaMalloc device arrays
    __half* device_matA = 0;
    __half* device_matBT = 0;
    __half* device_matC = 0;
    cudaMalloc((void**)&device_matA, M * K * sizeof(__half));
    cudaMalloc((void**)&device_matBT, N * K * sizeof(__half));
    cudaMalloc((void**)&device_matC, M * N * sizeof(__half));
    if(device_matA == 0 || device_matBT == 0 || device_matC == 0) {
        printf("couldn't allocate memory\n");
        return matC;
    }
    // cuda mem copy
    cudaMemcpy(device_matA, matA.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(device_matBT, matBT.data(), N * K * sizeof(__half), cudaMemcpyHostToDevice);
    // kernel call
    // note that the size is 2048 * 512, we choose 32 * 32 kernels
    int block_size = 32 * 32;
    int grid_size = (M * N) / block_size;
    start_timer(&timer);
    matrix_mul_smit_kernel_32x32<<<grid_size, block_size, 0>>>(device_matA, device_matBT, device_matC, M, N, K);
    float kernel_time_ms = stop_timer(&timer);
    cudaMemcpy(matC.data(), device_matC, M * N * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaFree(device_matA);
    cudaFree(device_matBT);
    cudaFree(device_matC);
    printf("cuda kernel <matrix_mul_smit_host> runtime %f ms.\n", kernel_time_ms);
    return matC;
}
