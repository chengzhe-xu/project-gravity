#include "cuda_utils.h"
#include "mat_trans.cuh"

__global__ void mat_trans_naive_kernel(float * mat, float * matT, int M, int N) {
    const unsigned int data_id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int row = data_id / N;
    unsigned int col = data_id % N;
    unsigned int transposed_data_id = col * M + row;
    matT[transposed_data_id] = mat[data_id];
    return;
}

matrix_template mat_trans_naive_host(matrix_template& mat, matrix_template& matT, int M, int N){
    event_pair timer;
    float * device_mat = 0;
    float * device_matT = 0;
    cudaMalloc((void**)&device_mat, M*N*sizeof(float));
    cudaMalloc((void**)&device_matT, N*M*sizeof(float));
    if (device_mat == 0 || device_matT == 0) {
        printf("[ERROR] Could not allocate CUDA memory.");
        return matT;
    }
    cudaMemcpy(device_mat, mat.data(), M*N*sizeof(float), cudaMemcpyHostToDevice);
    start_timer(&timer);
    int block_num = M*N / 512;
    int thread_num = 512;
    mat_trans_naive_kernel<<<block_num, thread_num>>>(device_mat, device_matT, M, N);
    float kernel_time_ms = stop_timer(&timer);
    cudaMemcpy(matT.data(), device_matT, N*M*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_mat);
    cudaFree(device_matT);
    printf("cuda kernel <mat_trans_naive_kernel> runtime %f ms.\n", kernel_time_ms);
    return matT;
}