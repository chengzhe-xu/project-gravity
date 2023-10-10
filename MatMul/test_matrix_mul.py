import numpy as np
import build.matrix_mul_lib as matrix_mul_lib

if __name__=='__main__':
    M, K, N = 2048*32, 128, 512
    test_round = 10
    matA = (np.random.rand(M, K) - 0.5) * 2 * 10
    matB = (np.random.rand(K, N) - 0.5) * 2 * 10
    matC = 0.0*(np.random.rand(M, N) - 0.5) * 2 * 10
    matA = np.ascontiguousarray(matA.astype(np.float16))
    matB = np.ascontiguousarray(matB.astype(np.float16))
    matC = np.ascontiguousarray(matC.astype(np.float16))

    matC_ref = np.matmul(matA, matB) + matC
    for _ in range(test_round):
        matC_naive_exp = matrix_mul_lib.mat_mul_naive(matA.T, matB, matC.copy(), M, N, K)
        print("Maximum abs error: ", np.max(np.abs(matC_naive_exp - matC_ref)))
    for _ in range(test_round):
        matC_simt_exp = matrix_mul_lib.mat_mul_simt(matA, matB.T, matC.copy(), M, N, K)
        print("Maximum abs error: ", np.max(np.abs(matC_simt_exp - matC_ref)))