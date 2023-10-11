#include <stdio.h>
#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <eigen3/Eigen/Dense>

__global__ void cast_kernel_float2half(__half* arr_h, float* arr, const int arr_size);
__global__ void cast_kernel_half2float(float* arr, __half* arr_h, const int arr_size);

using matrix_template = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
matrix_template matrix_mul_naive_host(const matrix_template& matAT, const matrix_template& matB, matrix_template& matC, int M, int N, int K);
matrix_template matrix_mul_half_host(const matrix_template& matAT, const matrix_template& matB, matrix_template& matC, int M, int N, int K);
matrix_template matrix_mul_smit_host(const matrix_template& matA, const matrix_template& matBT, matrix_template& matC, int M, int N, int K);