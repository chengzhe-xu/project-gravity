#include "cuda_utils.h"
#include "global_sum.cuh"

__global__ void global_sum_reduce_async_kernel(float * arr, float * sum_result) {
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

float global_sum_reduce_async_host(matrix_template& arr, const int arr_size) {
    event_pair overall_timer;
    float * arr_device = 0;
    float * sum_result_device = 0;
    float * arr_host = 0;
    const unsigned int stream_num = 8;
    cudaMalloc((void**)&arr_device, arr_size * sizeof(float));
    cudaMalloc((void**)&sum_result_device, stream_num * sizeof(float));
    cudaMallocHost((void**)&arr_host, arr_size * sizeof(float));
    if (arr_device == 0 || sum_result_device == 0 || arr_host == 0) {
        printf("[ERROR] Could not allocate CUDA memory or pinned memory.");
        return 0.0;
    }
    for (int i=0; i<arr_size; ++i) arr_host[i] = arr.data()[i];
    
    cudaStream_t streams[stream_num];
    float sum_result_host[stream_num];
    const unsigned int stream_chrunk_size = arr_size / stream_num;
    for (int stream_i=0; stream_i<stream_num; ++stream_i) {
        cudaStreamCreate(&streams[stream_i]);
    }
    const unsigned int block_num = stream_chrunk_size/(512*2);
    const unsigned int thread_num = 512;
    start_timer(&overall_timer);
    for (int stream_i=0; stream_i<stream_num; ++stream_i) {
        cudaMemcpyAsync(arr_device+stream_i*stream_chrunk_size, 
                        arr_host+stream_i*stream_chrunk_size, 
                        stream_chrunk_size*sizeof(float), 
                        cudaMemcpyHostToDevice,
                        streams[stream_i]);
        global_sum_reduce_async_kernel<<<block_num, thread_num, 0, streams[stream_i]>>>(arr_device+stream_i*stream_chrunk_size, sum_result_device+stream_i);
    }
    cudaMemcpy(sum_result_host, sum_result_device, stream_num * sizeof(float), cudaMemcpyDeviceToHost);
    for (int stream_i=1; stream_i<stream_num; ++stream_i) sum_result_host[0] += sum_result_host[stream_i];
    float host_time_ms = stop_timer(&overall_timer);
    for (int stream_i=0; stream_i<stream_num; ++stream_i) {
        cudaStreamDestroy(streams[stream_i]);
    } // destroy streams
    cudaFree(arr_device);
    cudaFree(sum_result_device);
    cudaFreeHost(arr_host); // free host buffer
    printf("cuda kernel <global_sum_reduce_async_kernel> host side runtime %f ms.\n", host_time_ms);
    return sum_result_host[0];
}