#include "matrix_mul.cuh"
#include "cuda_utils.h"

// We are not using A.T here, since in practice, A is feature, and might not be that convinient to get its transpose
#define LDG2S(a_share, b_share) \
{ \
    __half2 tmp_a[4], tmp_b[4]; \
    ldg128(from_a, tmp_a[0], tmp_a[1], tmp_a[2], tmp_a[3]); \
    ldg128(from_b, tmp_b[0], tmp_b[1], tmp_b[2], tmp_b[3]); \
    _Pragma("unroll") \
    for (int i=0; i<4; ++i){ \
        (a_share+to_As+i*2*(128+LD_buffer))[0] = tmp_a[i].x; \
        (a_share+to_As+(i*2+1)*(128+LD_buffer))[0] = tmp_a[i].y; \
    } \
    sts128(b_share+to_Bs, tmp_b[0], tmp_b[1], tmp_b[2], tmp_b[3]); \
    from_a += 8; from_b += (16*N)/2; \
} \

// #define LDG2S(a_share, b_share) \
// { \
//     __half2 tmp_a[4]; \
//     ldg128(from_a, tmp_a[0], tmp_a[1], tmp_a[2], tmp_a[3]); \
//     _Pragma("unroll") \
//     for (int i=0; i<4; ++i){ \
//         (a_share+to_As+i*2*(128+LD_buffer))[0] = tmp_a[i].x; \
//         (a_share+to_As+(i*2+1)*(128+LD_buffer))[0] = tmp_a[i].y; \
//     } \
//     ldgsts128(b_share+to_Bs, from_b); \
//     from_a += 8; from_b += (16*N)/2; \
// } \

#define MATMUL_WMMA(a_share, b_share) \
{ \
    nvcuda::wmma::load_matrix_sync(frag_a[0], a_share+warp_row*32, 128+LD_buffer); \
    nvcuda::wmma::load_matrix_sync(frag_a[1], a_share+warp_row*32+16, 128+LD_buffer); \
    _Pragma("unroll") \
    for (int i=0; i<4; ++i) { \
        nvcuda::wmma::load_matrix_sync(frag_b[i], b_share+warp_col*64+16*i, 128+LD_buffer); \
        nvcuda::wmma::mma_sync(frag_acc[0][i], frag_a[0], frag_b[i], frag_acc[0][i]); \
        nvcuda::wmma::mma_sync(frag_acc[1][i], frag_a[1], frag_b[i], frag_acc[1][i]); \
    } \
} \

// #define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var))) from cuda_fp16.hpp
__device__ __forceinline__ void ldg64(const __half2* addr, __half2 &reg0, __half2 &reg1){
    unsigned int reg0_ui, reg1_ui;
    asm volatile(
        "ld.global.nc.v2.b32 {%0, %1}, [%2];\n"
        : "=r"(reg0_ui),
          "=r"(reg1_ui)
        : "l"(addr)
    );
    reg0 = *(reinterpret_cast<__half2 *>(&reg0_ui));
    reg1 = *(reinterpret_cast<__half2 *>(&reg1_ui));
}

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

__device__ __forceinline__ void sts128(__half* addr, __half2 &reg0, __half2 &reg1, __half2 &reg2, __half2 &reg3){
    __half2* addr_shared_state = reinterpret_cast<__half2 *>(__cvta_generic_to_shared(addr));
    asm volatile(
        "st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
        :
        : "l"(addr_shared_state),
          "r"(*(reinterpret_cast<unsigned int *>(&reg0))),
          "r"(*(reinterpret_cast<unsigned int *>(&reg1))),
          "r"(*(reinterpret_cast<unsigned int *>(&reg2))),
          "r"(*(reinterpret_cast<unsigned int *>(&reg3)))
    );
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

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async
__device__ __forceinline__ void ldgsts128(__half* shared_addr, __half2* global_addr) {
    __half2* addr_shared_state = reinterpret_cast<__half2 *>(__cvta_generic_to_shared(shared_addr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16, 16;\n"
        :
        : "l"(addr_shared_state), 
          "l"(global_addr)
    );
}

// The executing thread can then use cp.async.wait_all or cp.async.wait_group or mbarrier instructions to wait for completion of 
// the asynchronous copy operation.
// No other synchronization mechanisms described in Memory Consistency Model can be used to guarantee the completion of the asynchronous copy operations.
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all
__device__ __forceinline__ void ldgsts_sync() {
    asm volatile("cp.async.wait_all ;\n" : :);
}

/*
This implementation is the SIMT core version.
For each block, we assign 16*16 threads,
For each thread, we assign 8*8 C matrix
For each block, we assign 128*128 C matrix,
For each warp, we assign 32*64 C matrix
For each step, we set k = 16
*/

/*
to debug matrix mul, it is very useful to use "cycle matrix" as input, 
0 1 2 3 4 5 6 7 8 9 0 1
2 3 4 5 6 7 8 9 0 1 2 3
.....
.....
4 5 6 7 8 9 0 1 2 3 4 5
and print out every details in the matmul process, 
mem bias, mem content, pA, pB, and check whether or not the output match expectation
*/

__global__ void matrix_mul_cpasync_kernel_128x128(__half2* matA, __half2* matB, __half2* matC, int M, int N, int K) {
    const unsigned int block_id = blockIdx.x;
    const unsigned int thread_id = threadIdx.x;
    const unsigned int block_row = block_id / (N/128);
    const unsigned int block_col = block_id % (N/128);
    const unsigned int warp_id = thread_id / 32;
    const unsigned int warp_row = warp_id / 2;
    const unsigned int warp_col = warp_id % 2;
    const unsigned int thread_row = (thread_id % 32) / 8;
    const unsigned int thread_col = (thread_id % 32) % 8;

    using fragA_t = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::col_major>;
    using fragB_t = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major>;
    using fragAcc_t = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half>;

    fragA_t frag_a[2];
    fragB_t frag_b[4];
    fragAcc_t frag_acc[2][4];
    __half* matC_h = reinterpret_cast<__half *>(matC) + (block_row*128 + warp_row*32) * N + block_col*128 + warp_col*64;
    #pragma unroll
    for (int i=0; i<2; ++i) {
        #pragma unroll
        for (int j=0; j<4; ++j) {
            // nvcuda::wmma::fill_fragment(frag_acc[i][j], __half(0.0));
            nvcuda::wmma::load_matrix_sync(frag_acc[i][j], matC_h + i*16*N + j*16, N, nvcuda::wmma::mem_row_major);
        }
    }

    const unsigned int LD_buffer = 16;

    // shared memory
    __shared__ __align__(4 * 1024) char smem[20 * 1024];
    // As/Bs needs 128 * 16 * half = 128 * 16 * 16 bits = 32768 bits = 32768 / 8 char = 4096 char
    // add the LD_buffer: need 4352 char = 4.25 k ==> 4.5 k
    // Cs needs 128 * (128 + LD_buffer) * half = 36864 = 36 k --- now we do not use share memory as a intermedia for acc
    __half* As[2] = {reinterpret_cast<__half *>(smem),
                    reinterpret_cast<__half *>(smem + 5120)};
    __half* Bs[2] = {reinterpret_cast<__half *>(smem + 5120*2),
                    reinterpret_cast<__half *>(smem + 5120*3)};
    // TODO: what is the __align__ used for and why we add some buffer into the share memory?

    // set the outer for loop initial value
    __half2* from_a = matA + (block_row*128 + (thread_id/2)) * (K/2) + 4*(thread_id%2);
    __half2* from_b = matB + (thread_id/16) * (N/2) + block_col*(128/2) + 4*(thread_id%16); 
    unsigned int to_As = (thread_id%2) * 8 * (128+LD_buffer) + (thread_id/2);
    unsigned int to_Bs = (thread_id/16) * (128+LD_buffer) + 8 * (thread_id%16);
    // outer loop
    LDG2S(As[0], Bs[0])
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[1], Bs[1])
    MATMUL_WMMA(As[0], Bs[0])
    // #pragma unroll
    // // for (int i_step=0; i_step<(K/16)/2-1; ++i_step) {
    // // for (int i_step=0; i_step<15; ++i_step) {
    //     ldgsts_sync();
    //     __syncthreads();
    //     LDG2S(As[0], Bs[0])
    //     MATMUL_WMMA(As[1], Bs[1])
    //     ldgsts_sync();
    //     __syncthreads();
    //     LDG2S(As[1], Bs[1])
    //     MATMUL_WMMA(As[0], Bs[0])
    // }
    // Why we finally choose to unroll the for loop by hand? What is the best practice to write double-buffer pipeline code?
    // Always keep this in mind: compiler is NOT that powerful when unrolling the for loop
    // it is hard to predict the dependency when unroll the for loop
    // the init implementation using arithmetic op, which need ALU in runtime, and hard for compiler to predict the results in compile time
    // so compiler can NOT schedule them in pipeline way
    // use logical ops is better, because it's simpler and easier to optimized in compile time
    // the best practice is to make the for loop (in-side code and the for conditions) as simple as possible
    // or just unroll it by hand
    // for K = 512, loop (512/16)/2-1 = 15 times
    // 1
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[0], Bs[0])
    MATMUL_WMMA(As[1], Bs[1])
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[1], Bs[1])
    MATMUL_WMMA(As[0], Bs[0])
    // 2
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[0], Bs[0])
    MATMUL_WMMA(As[1], Bs[1])
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[1], Bs[1])
    MATMUL_WMMA(As[0], Bs[0])
    // 3
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[0], Bs[0])
    MATMUL_WMMA(As[1], Bs[1])
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[1], Bs[1])
    MATMUL_WMMA(As[0], Bs[0])
    // 4
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[0], Bs[0])
    MATMUL_WMMA(As[1], Bs[1])
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[1], Bs[1])
    MATMUL_WMMA(As[0], Bs[0])
    // 5
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[0], Bs[0])
    MATMUL_WMMA(As[1], Bs[1])
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[1], Bs[1])
    MATMUL_WMMA(As[0], Bs[0])
    // 6
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[0], Bs[0])
    MATMUL_WMMA(As[1], Bs[1])
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[1], Bs[1])
    MATMUL_WMMA(As[0], Bs[0])
    // 7
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[0], Bs[0])
    MATMUL_WMMA(As[1], Bs[1])
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[1], Bs[1])
    MATMUL_WMMA(As[0], Bs[0])
    // 8
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[0], Bs[0])
    MATMUL_WMMA(As[1], Bs[1])
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[1], Bs[1])
    MATMUL_WMMA(As[0], Bs[0])
    // 9
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[0], Bs[0])
    MATMUL_WMMA(As[1], Bs[1])
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[1], Bs[1])
    MATMUL_WMMA(As[0], Bs[0])
    // 10
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[0], Bs[0])
    MATMUL_WMMA(As[1], Bs[1])
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[1], Bs[1])
    MATMUL_WMMA(As[0], Bs[0])
    // 11
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[0], Bs[0])
    MATMUL_WMMA(As[1], Bs[1])
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[1], Bs[1])
    MATMUL_WMMA(As[0], Bs[0])
    // 12
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[0], Bs[0])
    MATMUL_WMMA(As[1], Bs[1])
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[1], Bs[1])
    MATMUL_WMMA(As[0], Bs[0])
    // 13
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[0], Bs[0])
    MATMUL_WMMA(As[1], Bs[1])
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[1], Bs[1])
    MATMUL_WMMA(As[0], Bs[0])
    // 14
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[0], Bs[0])
    MATMUL_WMMA(As[1], Bs[1])
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[1], Bs[1])
    MATMUL_WMMA(As[0], Bs[0])
    // 15
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[0], Bs[0])
    MATMUL_WMMA(As[1], Bs[1])
    // ldgsts_sync();
    __syncthreads();
    LDG2S(As[1], Bs[1])
    MATMUL_WMMA(As[0], Bs[0])
    
    // ldgsts_sync();
    __syncthreads();
    MATMUL_WMMA(As[1], Bs[1])
    #pragma unroll
    for (int i=0; i<2; ++i) {
        #pragma unroll
        for (int j=0; j<4; ++j) {
            nvcuda::wmma::store_matrix_sync(matC_h + i*16*N + j*16, frag_acc[i][j], N, nvcuda::wmma::mem_row_major);
        }
    }
    __syncthreads();
    return;
}

matrix_template matrix_mul_cpasync_host(const matrix_template& matA, const matrix_template& matB, matrix_template& matC, int M, int N, int K) {
    event_pair timer;
    // cudaMalloc device arrays
    float* device_matA = 0;
    float* device_matB = 0;
    float* device_matC = 0;
    cudaMalloc((void**)&device_matA, M * K * sizeof(float));
    cudaMalloc((void**)&device_matB, K * N * sizeof(float));
    cudaMalloc((void**)&device_matC, M * N * sizeof(float));
    if(device_matA == 0 || device_matB == 0 || device_matC == 0) {
        printf("couldn't allocate memory\n");
        return matC;
    }
    // __half_copy
    __half* device_matA_h = 0;
    __half* device_matB_h = 0;
    __half* device_matC_h = 0;
    cudaMalloc((void**)&device_matA_h, M * K * sizeof(__half));
    cudaMalloc((void**)&device_matB_h, K * N * sizeof(__half));
    cudaMalloc((void**)&device_matC_h, M * N * sizeof(__half));
    if(device_matA_h == 0 || device_matB_h == 0 || device_matC_h == 0) {
        printf("couldn't allocate memory\n");
        return matC;
    }
    // cuda mem copy
    cudaMemcpy(device_matA, matA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_matB, matB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_matC, matC.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    cast_kernel_float2half<<<128, 256>>>(device_matA_h, device_matA, M * K);
    cast_kernel_float2half<<<128, 256>>>(device_matB_h, device_matB, K * N);
    cast_kernel_float2half<<<128, 256>>>(device_matC_h, device_matC, M * N);

    __half2* device_matA_h2 = reinterpret_cast<__half2 *>(device_matA_h); 
    __half2* device_matB_h2 = reinterpret_cast<__half2 *>(device_matB_h);
    __half2* device_matC_h2 = reinterpret_cast<__half2 *>(device_matC_h);

    // kernel call
    int block_size = 16 * 16;
    int grid_size = (M * N) / (128 * 128);
    start_timer(&timer);
    matrix_mul_cpasync_kernel_128x128<<<grid_size, block_size>>>(device_matA_h2, device_matB_h2, device_matC_h2, M, N, K);
    float kernel_time_ms = stop_timer(&timer);
    device_matC_h = reinterpret_cast<__half *>(device_matC_h2);
    cast_kernel_half2float<<<128, 256>>>(device_matC, device_matC_h, M * N);
    cudaMemcpy(matC.data(), device_matC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_matA);
    cudaFree(device_matB);
    cudaFree(device_matC);
    cudaFree(device_matA_h);
    cudaFree(device_matB_h);
    cudaFree(device_matC_h);
    printf("cuda kernel <matrix_mul_cpasync_kernel_128x128> runtime %f ms.\n", kernel_time_ms);
    return matC;
}
