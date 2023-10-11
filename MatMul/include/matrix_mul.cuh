#include <stdio.h>
#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <eigen3/Eigen/Dense>

__global__ void cast_kernel_float2half(__half* arr_h, float* arr, const int arr_size);
__global__ void cast_kernel_half2float(float* arr, __half* arr_h, const int arr_size);

__device__ __forceinline__ void ldg128(const __half2* addr, __half2 &reg0, __half2 &reg1, __half2 &reg2, __half2 &reg3);
__device__ __forceinline__ void stg128(__half2* addr, __half2 &reg0, __half2 &reg1, __half2 &reg2, __half2 &reg3);
__device__ __forceinline__ void half2matmulacc(__half2 acc[8][4], __half2 pA[8], __half2 pB[8]);

using matrix_template = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
matrix_template matrix_mul_naive_host(const matrix_template& matAT, const matrix_template& matB, matrix_template& matC, int M, int N, int K);
matrix_template matrix_mul_half_host(const matrix_template& matAT, const matrix_template& matB, matrix_template& matC, int M, int N, int K);
matrix_template matrix_mul_smit_host(const matrix_template& matA, const matrix_template& matBT, matrix_template& matC, int M, int N, int K);