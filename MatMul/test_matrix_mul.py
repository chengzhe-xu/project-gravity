import numpy as np
import build.matrix_mul_lib as matrix_mul_lib

def abs_error(mat, mat_ref):
    return np.max(np.abs(mat - mat_ref))

def rel_error(mat, mat_ref, esp=1e-9):
    return np.max(np.abs((mat - mat_ref) / (mat_ref + esp)))

def abs_error_w_rel(mat_2d, mat_ref_2d, esp=1e-9):
    mat = mat_2d.reshape([1, -1])[0]
    mat_ref = mat_ref_2d.reshape([1, -1])[0]
    abs_err = np.max(np.abs(mat - mat_ref))
    idx = np.argwhere(np.abs(mat - mat_ref) == abs_err).reshape([1, -1])[0]
    abs_rel = np.max(np.abs((mat[idx] - mat_ref[idx]) / (mat_ref[idx] + esp)))
    rel_err = np.max(np.abs((mat - mat_ref) / (mat_ref + esp)))
    return abs_err, abs_rel, rel_err

if __name__=='__main__':
    M, K, N = 2048, 128, 512
    test_round = 10
    np.random.seed(702)
    for _ in range(test_round):
        matA = (np.random.rand(M, K) - 0.5) * 2 * 10
        matB = (np.random.rand(K, N) - 0.5) * 2 * 10
        matC = (np.random.rand(M, N) - 0.5) * 2 * 10
        matA = np.ascontiguousarray(matA.astype(np.float32))
        matB = np.ascontiguousarray(matB.astype(np.float32))
        matC = np.ascontiguousarray(matC.astype(np.float32))

        matC_ref = np.matmul(matA.copy(), matB.copy()) + matC.copy()
        matC_naive_exp = matrix_mul_lib.mat_mul_naive(matA.T, matB, matC.copy(), M, N, K)
        naive_abs_err, naive_abs_rel, naive_rel_err = abs_error_w_rel(matC_naive_exp.copy(), matC_ref.copy())
        matC_half_exp = matrix_mul_lib.mat_mul_half(matA.T.astype(np.float16), matB.astype(np.float16), matC.copy().astype(np.float16), M, N, K)
        half_abs_err, half_abs_rel, half_rel_err = abs_error_w_rel(matC_half_exp.copy(), matC_ref.astype(np.float16).copy())
        print(f"Maximum abs error: naive version: abs: {naive_abs_err} ({100*naive_abs_rel}%)\
              half version: abs: {half_abs_err} ({100*half_abs_rel}%)")