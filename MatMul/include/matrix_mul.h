#include <stdio.h>
#include <cstdlib>
#include <cstdio>
#include <cuda.h>

void matrix_mul_naive_kernel_32x32(float* matA, float* matB, float* matC, int M, int N, int K);
void matrix_mul_naive_host(const float* matA, const float* matB, float* matC, int M, int N, int K);