import numpy as np
import build.matrix_mul_lib as matrix_mul_lib

if __name__=='__main__':
    # 2048 * 512, k=128
    matA = (np.random.rand(2048, 128) - 0.5) * 2 * 10
    matB = (np.random.rand(128, 512) - 0.5) * 2 * 10
    matC = 0.0*(np.random.rand(2048, 512) - 0.5) * 2 * 10
    matA = np.ascontiguousarray(matA).astype(float)
    matB = np.ascontiguousarray(matB).astype(float)
    matC = np.ascontiguousarray(matC).astype(float)

    matC_ref = np.matmul(matA, matB) + matC
    matC_exp = matrix_mul_lib.mat_mul_naive(matA, matB, matC, 2048, 512, 128)
    print("Maximum abs error: ", np.max(np.abs(matC_exp - matC_ref)))