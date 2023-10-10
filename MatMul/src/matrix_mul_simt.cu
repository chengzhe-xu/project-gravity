#include "matrix_mul.h"
#include "cuda_utils.h"

__global__ void matrix_mul_smit_kernel_32x32(float* matA, float* matBT, float* matC, int M, int N, int K) {
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
        // C[C_row, C_col] += A[C_row, i] * BT[C_col, i]
        tmp_C += matA[C_row * K+ i] * matBT[C_col * K + i];
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
    const uint64_t num_stream = 32;
    cudaStream_t streams[num_stream];
    int block_size = 32 * 32;
    int grid_size = (M * N) / (block_size * num_stream);
    for (int i_stream=0; i_stream<num_stream; ++i_stream) {
        cudaStreamCreate(&streams[i_stream]);
    }
    start_timer(&timer);
    for (int i_stream=0; i_stream<num_stream; ++i_stream) {
        matrix_mul_smit_kernel_32x32<<<grid_size, block_size, 0, streams[i_stream]>>>
        (device_matA+M*K*i_stream/num_stream, device_matBT, device_matC+M*N*i_stream/num_stream, M/num_stream, N, K);
    }
    for (int i_stream=0; i_stream<num_stream; ++i_stream) {
        cudaStreamSynchronize(streams[i_stream]);
    } // if we do not record time, this is not needed since cudaMemcpy is blocking op
    float kernel_time_ms = stop_timer(&timer);
    cudaMemcpy(matC.data(), device_matC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i_stream=0; i_stream<num_stream; ++i_stream) {
        cudaStreamDestroy(streams[i_stream]);
    }
    cudaFree(device_matA);
    cudaFree(device_matBT);
    cudaFree(device_matC);
    printf("cuda kernel <matrix_mul_smit_host> runtime %f ms.\n", kernel_time_ms);
    return matC;
}
