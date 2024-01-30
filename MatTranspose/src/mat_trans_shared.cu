#include "cuda_utils.h"
#include "mat_trans.cuh"

#define SWAP_HALF(a, b) \
{ \
    swap_tmp = a.y; \
    a.y = b.x; \
    b.x = swap_tmp; \
} \

__device__ __forceinline__ void ldg128(const __half2* addr, __half2 &reg0, __half2 &reg1, __half2 &reg2, __half2 &reg3){
    unsigned int reg0_ui, reg1_ui, reg2_ui, reg3_ui;
    asm volatile(
        "ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(reg0_ui),
          "=r"(reg1_ui),
          "=r"(reg2_ui),
          "=r"(reg3_ui)
        : "l"(addr)
    );
    reg0 = *(reinterpret_cast<__half2 *>(&reg0_ui));
    reg1 = *(reinterpret_cast<__half2 *>(&reg1_ui));
    reg2 = *(reinterpret_cast<__half2 *>(&reg2_ui));
    reg3 = *(reinterpret_cast<__half2 *>(&reg3_ui));
}

__device__ __forceinline__ void stg128(__half2* addr, __half2 &reg0, __half2 &reg1, __half2 &reg2, __half2 &reg3) {
    asm volatile(
        "st.global.v4.b32 [%0], {%1, %2, %3, %4};\n"
        :
        : "l"(addr),
          "r"(*(reinterpret_cast<unsigned int *>(&reg0))),
          "r"(*(reinterpret_cast<unsigned int *>(&reg1))),
          "r"(*(reinterpret_cast<unsigned int *>(&reg2))),
          "r"(*(reinterpret_cast<unsigned int *>(&reg3)))
    );
}

__device__ __forceinline__ void lds128(__half2* addr, __half2 &reg0, __half2 &reg1, __half2 &reg2, __half2 &reg3){
    unsigned int* addr_shared_state = reinterpret_cast<unsigned int *>(__cvta_generic_to_shared(addr));
    unsigned int reg0_ui, reg1_ui, reg2_ui, reg3_ui;
    asm volatile(
        "ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(reg0_ui),
          "=r"(reg1_ui),
          "=r"(reg2_ui),
          "=r"(reg3_ui)
        : "l"(addr_shared_state)
    );
    reg0 = *(reinterpret_cast<__half2 *>(&reg0_ui));
    reg1 = *(reinterpret_cast<__half2 *>(&reg1_ui));
    reg2 = *(reinterpret_cast<__half2 *>(&reg2_ui));
    reg3 = *(reinterpret_cast<__half2 *>(&reg3_ui));
}

__global__ void mat_trans_shared_kernel(__half2 * mat, __half2 * matT, int M, int N) {
    // launch 16*16 threads, handling 64*64 data
    unsigned int block_row = blockIdx.x / (N/64);
    unsigned int block_col = blockIdx.x % (N/64);
    unsigned int thread_row = threadIdx.x / 8;
    unsigned int thread_col = threadIdx.x % 8;
    const unsigned int LD_buffer = 1;

    __shared__ __align__(16 * 1024) char smem[8448];
    __half2* mat_s = reinterpret_cast<__half2 * >(smem);

    __half2 * from_mat  = mat + block_row*64*(N/2) + block_col*32 + thread_row*2*(N/2) + thread_col*4;

    __half2 tmp_mat[8];
    ldg128(from_mat, tmp_mat[0], tmp_mat[1], tmp_mat[2], tmp_mat[3]);
    ldg128(from_mat+(N/2), tmp_mat[4], tmp_mat[5], tmp_mat[6], tmp_mat[7]);
    __half swap_tmp;
    SWAP_HALF(tmp_mat[0], tmp_mat[4])
    SWAP_HALF(tmp_mat[1], tmp_mat[5])
    SWAP_HALF(tmp_mat[2], tmp_mat[6])
    SWAP_HALF(tmp_mat[3], tmp_mat[7])
    __half2 * to_mat_s = mat_s + thread_col*8*(32+LD_buffer) + thread_row;
    to_mat_s[0] = tmp_mat[0];
    to_mat_s[32+LD_buffer] = tmp_mat[4];
    to_mat_s[(32+LD_buffer)*2] = tmp_mat[1];
    to_mat_s[(32+LD_buffer)*3] = tmp_mat[5];
    to_mat_s[(32+LD_buffer)*4] = tmp_mat[2];
    to_mat_s[(32+LD_buffer)*5] = tmp_mat[6];
    to_mat_s[(32+LD_buffer)*6] = tmp_mat[3];
    to_mat_s[(32+LD_buffer)*7] = tmp_mat[7];
    __syncthreads();
    __half2 * from_mat_s = mat_s + thread_row*2*(32+LD_buffer) + thread_col*4;
    // lds128(from_mat_s, tmp_mat[0], tmp_mat[1], tmp_mat[2], tmp_mat[3]);
    // lds128(from_mat_s+(32+LD_buffer), tmp_mat[4], tmp_mat[5], tmp_mat[6], tmp_mat[7]);
    #pragma unroll
    for (int lds_idx=0; lds_idx<4; ++lds_idx) {
        tmp_mat[lds_idx] = from_mat_s[lds_idx];
        tmp_mat[lds_idx+4] = from_mat_s[lds_idx+(32+LD_buffer)];
    }
    __half2 * to_matT = matT + (block_row*32) + (block_col*64)*(M/2) + thread_row*2*(M/2) + thread_col*4;
    stg128(to_matT, tmp_mat[0], tmp_mat[1], tmp_mat[2], tmp_mat[3]);
    stg128(to_matT+(M/2), tmp_mat[4], tmp_mat[5], tmp_mat[6], tmp_mat[7]);
    return;
}

matrix_template mat_trans_shared_host(matrix_template& mat, matrix_template& matT, int M, int N){
    event_pair timer;
    float * device_mat = 0;
    float * device_matT = 0;
    __half * device_mat_h = 0;
    __half * device_matT_h = 0;
    cudaMalloc((void**)&device_mat, M*N*sizeof(float));
    cudaMalloc((void**)&device_mat_h, M*N*sizeof(__half));
    cudaMalloc((void**)&device_matT, N*M*sizeof(float));
    cudaMalloc((void**)&device_matT_h, N*M*sizeof(__half));
    if (device_mat == 0 || device_matT == 0 || device_mat_h == 0 || device_matT_h == 0) {
        printf("[ERROR] Could not allocate CUDA memory.");
        return matT;
    }
    cudaMemcpy(device_mat, mat.data(), M*N*sizeof(float), cudaMemcpyHostToDevice);
    cast_kernel_float2half<<<128, 256>>>(device_mat_h, device_mat, M*N);
    __half2 * device_mat_h2 = reinterpret_cast<__half2 * >(device_mat_h);
    __half2 * device_matT_h2 = reinterpret_cast<__half2 * >(device_matT_h);
    start_timer(&timer);
    int block_num = M*N / (64*64);
    int thread_num = 256;
    mat_trans_shared_kernel<<<block_num, thread_num>>>(device_mat_h2, device_matT_h2, M, N);
    float kernel_time_ms = stop_timer(&timer);
    device_matT_h = reinterpret_cast<__half * >(device_matT_h2);
    cast_kernel_half2float<<<128, 256>>>(device_matT, device_matT_h, N*M);
    cudaMemcpy(matT.data(), device_matT, N*M*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_mat);
    cudaFree(device_matT);
    cudaFree(device_mat_h);
    cudaFree(device_matT_h);
    printf("cuda kernel <mat_trans_shared_kernel> runtime %f ms.\n", kernel_time_ms);
    return matT;
}