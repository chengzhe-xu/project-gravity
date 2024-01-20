#include "cuda_utils.h"
#include "global_sum.cuh"

__global__ void global_sum_reduce_kernel(float * arr, float * sum_result) {
    // 512 threads per block, 1024 data per block
    unsigned int arr_size = 1024;
    const unsigned int data_id = blockIdx.x * arr_size + threadIdx.x;

    __shared__ __align__(2 * 1024) char smem[512 * 4]; // 2k
    float * arr_s = reinterpret_cast<float *>(smem);
    // load global
    arr_s[threadIdx.x] = arr[data_id] + arr[data_id + arr_size / 2];
    arr_size /= 2;
    __syncthreads();
    while (arr_size > 1) {
        if (threadIdx.x < arr_size / 2) arr_s[threadIdx.x] += arr_s[threadIdx.x + arr_size / 2];
        arr_size /= 2;
        __syncthreads();
    }
    // the result is in arr_s[0];
    if (threadIdx.x == 0) atomicAdd(sum_result, arr_s[0]);
    return;
}

float global_sum_reduce_host(matrix_template& arr, const int arr_size) {
    event_pair timer;
    event_pair overall_timer;
    float * arr_device = 0;
    float * sum_result_device = 0;
    cudaMalloc((void**)&arr_device, arr_size * sizeof(float));
    cudaMalloc((void**)&sum_result_device, 1 * sizeof(float));
    if (arr_device == 0 || sum_result_device == 0) {
        printf("[ERROR] Could not allocate CUDA memory.");
        return 0.0;
    }
    start_timer(&overall_timer);
    cudaMemcpy(arr_device, arr.data(), arr_size * sizeof(float), cudaMemcpyHostToDevice);
    start_timer(&timer);
    const unsigned int block_num = arr_size/(512*2);
    const unsigned int thread_num = 512;
    global_sum_reduce_kernel<<<block_num, thread_num>>>(arr_device, sum_result_device);
    float kernel_time_ms = stop_timer(&timer);
    float sum_result_host = 0;
    cudaMemcpy(&sum_result_host, sum_result_device, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    float host_time_ms = stop_timer(&overall_timer);
    cudaFree(arr_device);
    cudaFree(sum_result_device);
    printf("cuda kernel <global_sum_reduce_kernel> runtime %f ms, host side runtime %f ms.\n", kernel_time_ms, host_time_ms);
    return sum_result_host;
}