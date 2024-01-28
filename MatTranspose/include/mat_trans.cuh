#include <stdio.h>
#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <eigen3/Eigen/Dense>
#include <mma.h>

__global__ void cast_kernel_float2half(__half* arr_h, float* arr, const int arr_size);
__global__ void cast_kernel_half2float(float* arr, __half* arr_h, const int arr_size);

using matrix_template = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
matrix_template mat_trans_naive_host(matrix_template& mat, matrix_template& matT, int M, int N);
matrix_template mat_trans_shared_host(matrix_template& mat, matrix_template& matT, int M, int N);