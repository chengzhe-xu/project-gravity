#include "matrix_mul.h"
#include "cuda_utils.h"

__global__ void matrix_mul_smit_kernel_32x32(float* matA, float* matBT, float* matC, int M, int N, int K) {
    // TODO
    return;
}

matrix_template matrix_mul_smit_host(const matrix_template& matA, const matrix_template& matBT, matrix_template& matC, int M, int N, int K) {
    event_pair timer;
    // cudaMalloc device arrays
    float* device_matA = 0;
    float* device_matBT = 0;
    float* device_matC = 0;
    cudaMalloc((void**)&device_matA, M * K * sizeof(float));
    cudaMalloc((void**)&device_matBT, N * K * sizeof(float));
    cudaMalloc((void**)&device_matC, M * N * sizeof(float));
    if(device_matA == 0 || device_matBT == 0 || device_matC == 0) {
        printf("couldn't allocate memory\n");
        return matC;
    }
    // cuda mem copy
    cudaMemcpy(device_matA, matA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_matBT, matBT.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);
    // kernel call
    // note that the size is 2048 * 512, we choose 32 * 32 kernels
    int block_size = 32 * 32;
    int grid_size = (M * N) / block_size;
    start_timer(&timer);
    matrix_mul_smit_kernel_32x32<<<grid_size, block_size>>>(device_matA, device_matBT, device_matC, M, N, K);
    float kernel_time_ms = stop_timer(&timer);
    cudaMemcpy(matC.data(), device_matC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_matA);
    cudaFree(device_matBT);
    cudaFree(device_matC);
    printf("cuda kernel <matrix_mul_smit_host> runtime %f ms.\n", kernel_time_ms);
    return matC;
}
