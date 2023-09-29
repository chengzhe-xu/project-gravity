#include "matrix_mul.h"
#include "cuda_utils.h"

__global__ void matrix_mul_naive_kernel_32x32(float* matAT, float* matB, float* matC, int M, int N, int K) {
    const unsigned int block_id = blockIdx.x;
    const unsigned int thread_id = threadIdx.x;
    const unsigned int block_row = block_id / 16;
    const unsigned int block_col = block_id % 16;
    const unsigned int thread_row = thread_id / 32;
    const unsigned int thread_col = thread_id % 32;

    int C_row = 32 * block_row + thread_row;
    int C_col = 32 * block_col + thread_col;

    // fma
    float tmp_C = matC[N * C_row + C_col];
    for (int i=0; i<K; ++i) {
        // C[C_row, C_col] += AT[i, C_row] * B[i, C_col]
        tmp_C += matAT[M * i + C_row] * matB[N * i + C_col];
        // TODO: initially it was:
        // matC[N * C_row + C_col] += matA[K * C_row + i] * matB[N * i + C_col];
        // why the current one is better (2.28x faster)?
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
    // cuda mem copy
    cudaMemcpy(device_matAT, matAT.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_matB, matB.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);
    // kernel call
    // note that the size is 2048 * 512, we choose 32 * 32 kernels
    int block_size = 32 * 32;
    int grid_size = (M * N) / block_size;
    start_timer(&timer);
    matrix_mul_naive_kernel_32x32<<<grid_size, block_size>>>(device_matAT, device_matB, device_matC, M, N, K);
    float kernel_time_ms = stop_timer(&timer);
    cudaMemcpy(matC.data(), device_matC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_matAT);
    cudaFree(device_matB);
    cudaFree(device_matC);
    printf("cuda kernel <matrix_mul_naive_kernel_32x32> runtime %f ms.\n", kernel_time_ms);
    return matC;
}
