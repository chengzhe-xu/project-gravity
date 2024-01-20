#include <stdio.h>
#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <eigen3/Eigen/Dense>
#include <mma.h>

using matrix_template = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
float global_sum_atomAdd_host(matrix_template& arr, const int arr_size);
float global_sum_reduce_host(matrix_template& arr, const int arr_size);