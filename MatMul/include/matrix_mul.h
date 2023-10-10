#include <stdio.h>
#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <eigen3/Eigen/Dense>

using matrix_template = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
matrix_template matrix_mul_naive_host(const matrix_template& matA, const matrix_template& matB, matrix_template& matC, int M, int N, int K);
matrix_template matrix_mul_smit_host(const matrix_template& matA, const matrix_template& matBT, matrix_template& matC, int M, int N, int K);