#include "matrix_mul.cuh"
#include "cuda_utils.h"

// #define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var))) from cuda_fp16.hpp
__device__ __forceinline__ void ldg128(const __half2* addr, __half2 &reg0, __half2 &reg1, __half2 &reg2, __half2 &reg3){
    asm volatile(
        "ld.global.nc.v4.b32 {%1, %2, %3, %4}, [%0];\n"
        : "=r"(*(reinterpret_cast<unsigned int *>(&(reg0)))),
          "=r"(*(reinterpret_cast<unsigned int *>(&(reg1)))),
          "=r"(*(reinterpret_cast<unsigned int *>(&(reg2)))),
          "=r"(*(reinterpret_cast<unsigned int *>(&(reg3))))
        : "l"(addr)
    );
}

__device__ __forceinline__ void stg128(__half2* addr, __half2 &reg0, __half2 &reg1, __half2 &reg2, __half2 &reg3) {
    asm volatile(
        "st.global.v4.b32 [%0], {%1, %2, %3, %4};\n"
        :
        : "l"(addr),
          "r"(*(reinterpret_cast<unsigned int *>(&(reg0)))),
          "r"(*(reinterpret_cast<unsigned int *>(&(reg1)))),
          "r"(*(reinterpret_cast<unsigned int *>(&(reg2)))),
          "r"(*(reinterpret_cast<unsigned int *>(&(reg3))))
    );
}

__device__ __forceinline__ void half2matmulacc(__half2 acc[8][4], __half2 pA[8], __half2 pB[8]) {
    #pragma unroll
    for (int i=0; i<8; ++i) {
        #pragma unroll
        for (int j=0; j<4; ++j) {
            // acc[i][j].x += pA[i].x * pB[2*j].x + pA[i].y * pB[2*j].y;
            // acc[i][j].y += pA[i].x * pB[2*j+1].x + pA[i].y * pB[2*j+1].y;
            __half2 tmp[3] = {__half2{pB[2*j].x, pB[2*j+1].y},
                              __half2{pA[i].y, pA[i].x},
                              __half2{pB[2*j].y, pB[2*j+1].x}};
            acc[i][j] = __hfma2(pA[i], tmp[0], acc[i][j]);
            acc[i][j] = __hfma2(tmp[1], tmp[2], acc[i][j]);
        }
    }
    return;
}

/*
This implementation is the SIMT core version.
For each block, we assign 16*16 threads,
For each thread, we assign 8*8 C matrix
For each block, we assign 128*128 C matrix,
For each warp, we assign 32*64 C matrix
For each step, we set k = 16
*/

__global__ void matrix_mul_smit_kernel_128x128(__half2* matA, __half2* matBT, __half2* matC, int M, int N, int K) {
    const unsigned int block_id = blockIdx.x;
    const unsigned int thread_id = threadIdx.x;
    const unsigned int block_row = block_id / (N/128);
    const unsigned int block_col = block_id % (N/128);
    const unsigned int warp_id = thread_id / 32;
    const unsigned int warp_row = warp_id / 2;
    const unsigned int warp_col = warp_id % 2;
    const unsigned int thread_row = (thread_id % 32) / 8;
    const unsigned int thread_col = (thread_id % 32) % 8;

    const unsigned int LD_buffer = 8;

    // shared memory
    __shared__ __align__(16 * 1024) char smem[9 * 1024];
    // As/Bs needs 128 * 16 * half = 128 * 16 * 16 bits = 32768 bits = 32768 / 8 char = 4096 char
    // add the LD_buffer: need 4352 char = 4.25 k ==> 4.5 k
    __half2* As = reinterpret_cast<__half2 *>(smem);
    __half2* Bs = reinterpret_cast<__half2 *>(smem + 4608);
    // TODO: what is the __align__ used for and why we add some buffer into the share memory?

    __half2 acc[8][4];
    // load C into the acc
    __half2* from_c = matC + (block_row*128 + warp_row*32 + thread_row*8) * (N/2) + block_col*128/2 + warp_col*64/2 + thread_col*8/2;
    #pragma unroll
    for (int i=0; i<8; ++i) {
        // ldg128(from_c+i*N/2, acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        acc[i][0] = __ldg(from_c+i*N/2);
        acc[i][1] = __ldg(from_c+i*N/2 + 1);
        acc[i][2] = __ldg(from_c+i*N/2 + 2);
        acc[i][3] = __ldg(from_c+i*N/2 + 3);
    }
    // __syncthreads();

    // set the outer for loop initial value
    __half2* from_a = matA + (block_row*128 + 4*(thread_id/8)) * (K/2) + thread_id%8;
    __half2* to_As[4] = {As + (thread_id%8) * (128+LD_buffer) + (( 4*(thread_id/8)+0 )%8) * 16 + ( 4*(thread_id/8)+0 )/8,
                         As + (thread_id%8) * (128+LD_buffer) + (( 4*(thread_id/8)+1 )%8) * 16 + ( 4*(thread_id/8)+1 )/8,
                         As + (thread_id%8) * (128+LD_buffer) + (( 4*(thread_id/8)+2 )%8) * 16 + ( 4*(thread_id/8)+2 )/8,
                         As + (thread_id%8) * (128+LD_buffer) + (( 4*(thread_id/8)+3 )%8) * 16 + ( 4*(thread_id/8)+3 )/8};
    __half2* from_b = matBT + (block_col*128 + 4*thread_id/8) * (K/2) + thread_id%8; 
    __half2* to_Bs[4] = {Bs + (thread_id%8) * (128+LD_buffer) + (( 4*(thread_id/8)+0 )%8) * 16 + ( 4*(thread_id/8)+0 )/8,
                         Bs + (thread_id%8) * (128+LD_buffer) + (( 4*(thread_id/8)+1 )%8) * 16 + ( 4*(thread_id/8)+1 )/8,
                         Bs + (thread_id%8) * (128+LD_buffer) + (( 4*(thread_id/8)+2 )%8) * 16 + ( 4*(thread_id/8)+2 )/8,
                         Bs + (thread_id%8) * (128+LD_buffer) + (( 4*(thread_id/8)+3 )%8) * 16 + ( 4*(thread_id/8)+3 )/8};
    // TODO: maybe too much register occupied, bad for occupacy rate

    // outer loop
    #pragma unroll
    for (int i_step=0; i_step<K/16; ++i_step) {
        // load sub A matrix
        __half2 tmp_a[4] = {
            __ldg(from_a),
            __ldg(from_a + 1*K/2),
            __ldg(from_a + 2*K/2),
            __ldg(from_a + 3*K/2)
        };
        #pragma unroll
        for (int i=0; i<4; ++i) to_As[i][0] = tmp_a[i];
        // load sub B matrix
        __half2 tmp_b[4] = {
            __ldg(from_b),
            __ldg(from_b + 1*K/2),
            __ldg(from_b + 2*K/2),
            __ldg(from_b + 3*K/2),
        };
        #pragma unroll
        for (int i=0; i<4; ++i) to_Bs[i][0] = tmp_b[i];
        __syncthreads();
        from_a += 8; from_b += 8;
        // set the inner for loop initial value
        __half2* from_As[8] = {
            As + 16*((warp_row*32 + thread_row*8 + 0)%8) + (warp_row*32 + thread_row*8 + 0)/8,
            As + 16*((warp_row*32 + thread_row*8 + 1)%8) + (warp_row*32 + thread_row*8 + 1)/8,
            As + 16*((warp_row*32 + thread_row*8 + 2)%8) + (warp_row*32 + thread_row*8 + 2)/8,
            As + 16*((warp_row*32 + thread_row*8 + 3)%8) + (warp_row*32 + thread_row*8 + 3)/8,
            As + 16*((warp_row*32 + thread_row*8 + 4)%8) + (warp_row*32 + thread_row*8 + 4)/8,
            As + 16*((warp_row*32 + thread_row*8 + 5)%8) + (warp_row*32 + thread_row*8 + 5)/8,
            As + 16*((warp_row*32 + thread_row*8 + 6)%8) + (warp_row*32 + thread_row*8 + 6)/8,
            As + 16*((warp_row*32 + thread_row*8 + 7)%8) + (warp_row*32 + thread_row*8 + 7)/8,
        };
        __half2* from_Bs[8] = {
            Bs + 16*((warp_col*64 + thread_col*8 + 0)%8) + (warp_col*64 + thread_col*8 + 0)/8,
            Bs + 16*((warp_col*64 + thread_col*8 + 1)%8) + (warp_col*64 + thread_col*8 + 1)/8,
            Bs + 16*((warp_col*64 + thread_col*8 + 2)%8) + (warp_col*64 + thread_col*8 + 2)/8,
            Bs + 16*((warp_col*64 + thread_col*8 + 3)%8) + (warp_col*64 + thread_col*8 + 3)/8,
            Bs + 16*((warp_col*64 + thread_col*8 + 4)%8) + (warp_col*64 + thread_col*8 + 4)/8,
            Bs + 16*((warp_col*64 + thread_col*8 + 5)%8) + (warp_col*64 + thread_col*8 + 5)/8,
            Bs + 16*((warp_col*64 + thread_col*8 + 6)%8) + (warp_col*64 + thread_col*8 + 6)/8,
            Bs + 16*((warp_col*64 + thread_col*8 + 7)%8) + (warp_col*64 + thread_col*8 + 7)/8,
        };
        // TODO: maybe too much register occupied, bad for occupacy rate

        // inner loop
        #pragma unroll
        for (int i_inner_step=0; i_inner_step<8; ++i_inner_step) {
            __half2 pA[8], pB[8];
            #pragma unroll
            for (int i=0; i<8; ++i){
                pA[i] = from_As[i][0];
                pB[i] = from_Bs[i][0];
            }
            // bug: every odd inner lop, i_inner_step = 2, 4, 6...
            // pB pointer is right-switched by 1
            if (blockIdx.x == 9 && threadIdx.x == 75) {
                printf("i_step, %d, i_inner_step, %d\n", i_step, i_inner_step);
                // printf("pA: %f\n", __half2float(pA[0].x));
                // for (int vis_i=0; vis_i<8; ++vis_i){
                //     printf("%f, ", __half2float(pA[vis_i].x));
                // }
                // printf("\n");
                // for (int vis_i=0; vis_i<8; ++vis_i){
                //     printf("%f, ", __half2float(pA[vis_i].y));
                // }
                // printf("\n");
                printf("pB: %f\n", __half2float(pB[0].x));
                // for (int vis_i=0; vis_i<8; ++vis_i){
                //     printf("%f, ", __half2float(pB[vis_i].x));
                // }
                // printf("\n");
                // for (int vis_i=0; vis_i<8; ++vis_i){
                //     printf("%f, ", __half2float(pB[vis_i].y));
                // }
                printf("\n");
            }
            half2matmulacc(acc, pA, pB);
            #pragma unroll
            for (int i=0; i<8; ++i){
                from_As[i] += (128+LD_buffer);
                from_Bs[i] += (128+LD_buffer);
            }
            // __syncthreads();
        }
        __syncthreads();
    }
    // write back to C
    __half2* to_c = matC + (block_row*128 + warp_row*32 + thread_row*8) * (N/2) + block_col*128/2 + warp_col*64/2 + thread_col*8/2;
    #pragma unroll
    for (int i=0; i<8; ++i) {
        // stg128(to_c+i*N/2, acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        (to_c+i*N/2)[0] = acc[i][0];
        (to_c+i*N/2)[1] = acc[i][1];
        (to_c+i*N/2)[2] = acc[i][2];
        (to_c+i*N/2)[3] = acc[i][3];
    }
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
    // __half_copy
    __half* device_matA_h = 0;
    __half* device_matBT_h = 0;
    __half* device_matC_h = 0;
    cudaMalloc((void**)&device_matA_h, M * K * sizeof(__half));
    cudaMalloc((void**)&device_matBT_h, N * K * sizeof(__half));
    cudaMalloc((void**)&device_matC_h, M * N * sizeof(__half));
    if(device_matA_h == 0 || device_matBT_h == 0 || device_matC_h == 0) {
        printf("couldn't allocate memory\n");
        return matC;
    }
    // cuda mem copy
    cudaMemcpy(device_matA, matA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_matBT, matBT.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_matC, matC.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    cast_kernel_float2half<<<128, 256>>>(device_matA_h, device_matA, M * K);
    cast_kernel_float2half<<<128, 256>>>(device_matBT_h, device_matBT, N * K);
    cast_kernel_float2half<<<128, 256>>>(device_matC_h, device_matC, M * N);

    __half2* device_matA_h2 = reinterpret_cast<__half2 *>(device_matA_h); 
    __half2* device_matBT_h2 = reinterpret_cast<__half2 *>(device_matBT_h);
    __half2* device_matC_h2 = reinterpret_cast<__half2 *>(device_matC_h);

    // kernel call
    int block_size = 16 * 16;
    int grid_size = (M * N) / (128 * 128);
    start_timer(&timer);
    matrix_mul_smit_kernel_128x128<<<grid_size, block_size>>>(device_matA_h2, device_matBT_h2, device_matC_h2, M, N, K);
    float kernel_time_ms = stop_timer(&timer);
    device_matC_h = reinterpret_cast<__half *>(device_matC_h2);
    cast_kernel_half2float<<<128, 256>>>(device_matC, device_matC_h, M * N);
    cudaMemcpy(matC.data(), device_matC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_matA);
    cudaFree(device_matBT);
    cudaFree(device_matC);
    cudaFree(device_matA_h);
    cudaFree(device_matBT_h);
    cudaFree(device_matC_h);
    printf("cuda kernel <matrix_mul_smit_kernel_128x128> runtime %f ms.\n", kernel_time_ms);
    return matC;
}
