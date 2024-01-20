#include "cuda_utils.h"
#include "global_sum.cuh"

__global__ void global_sum_atomAdd_kernel(float * arr, float * sum_result) {
    const unsigned int data_id = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(sum_result, arr[data_id]);
    return;
}

float global_sum_atomAdd_host(matrix_template& arr, const int arr_size) {
    event_pair timer;
    float * device_arr = 0;
    float * device_sum_result = 0;
    cudaMalloc((void**)&device_arr, arr_size * sizeof(float));
    cudaMalloc((void**)&device_sum_result, 1 * sizeof(float));
    if (device_arr == 0 || device_sum_result == 0) {
        printf("[ERROR] Could not allocate CUDA memory.");
        return 0.0;
    }
    cudaMemcpy(device_arr, arr.data(), arr_size * sizeof(float), cudaMemcpyHostToDevice);
    start_timer(&timer);
    int block_num = arr_size / 512;
    int thread_num = 512;
    global_sum_atomAdd_kernel<<<block_num, thread_num>>>(device_arr, device_sum_result);
    float kernel_time_ms = stop_timer(&timer);
    float sum_result_host = 0;
    cudaMemcpy(&sum_result_host, device_sum_result, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_arr);
    cudaFree(device_sum_result);
    printf("cuda kernel <global_sum_atomAdd_kernel> runtime %f ms.\n", kernel_time_ms);
    return sum_result_host;
}