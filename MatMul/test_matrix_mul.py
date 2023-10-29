import numpy as np
import argparse
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

def argument_parser():
    parser = argparse.ArgumentParser(description="Matrix multiplication arg parser.")
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--K", type=int, default=512)
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--test_round", type=int, default=5)
    parser.add_argument("--is_profile", type=bool, default=False)
    return parser.parse_args()

if __name__=='__main__':
    args = argument_parser()
    M, K, N = args.M, args.K, args.N
    np.random.seed(np.random.randint(7))
    for _ in range(args.test_round):
        matA = (np.random.rand(M, K) - 0.5) * 2 * 10
        matB = (np.random.rand(K, N) - 0.5) * 2 * 10
        matC = (np.random.rand(M, N) - 0.5) * 2 * 10
        matA = np.ascontiguousarray(matA.astype(np.float32))
        matB = np.ascontiguousarray(matB.astype(np.float32))
        matC = np.ascontiguousarray(matC.astype(np.float32))

        matC_ref = np.matmul(matA.copy(), matB.copy()) + matC.copy()

        matC_naive_exp = matrix_mul_lib.mat_mul_naive(matA.T, matB, matC.copy(), M, N, K)
        if not args.is_profile:
            naive_abs_err, naive_abs_rel, naive_rel_err = abs_error_w_rel(matC_naive_exp.copy(), matC_ref.copy())

        matC_half_exp = matrix_mul_lib.mat_mul_half(matA.T.astype(np.float16), matB.astype(np.float16), matC.copy().astype(np.float16), M, N, K)
        if not args.is_profile:
            half_abs_err, half_abs_rel, half_rel_err = abs_error_w_rel(matC_half_exp.copy(), matC_ref.astype(np.float16).copy())

        matC_simt_exp = matrix_mul_lib.mat_mul_simt(matA.astype(np.float16), matB.T.astype(np.float16), matC.copy().astype(np.float16), M, N, K)
        if not args.is_profile:
            simt_abs_err, simt_abs_rel, smit_rel_err = abs_error_w_rel(matC_simt_exp.copy(), matC_ref.astype(np.float16).copy())

        matC_simt_pipeline_exp = matrix_mul_lib.mat_mul_simt_pipeline(matA.astype(np.float16), matB.T.astype(np.float16), matC.copy().astype(np.float16), M, N, K)
        if not args.is_profile:
            simt_pipeline_abs_err, simt_pipeline_abs_rel, smit_pipeline_rel_err = abs_error_w_rel(matC_simt_pipeline_exp.copy(), matC_ref.astype(np.float16).copy())

        matC_tensorcore_exp = matrix_mul_lib.mat_mul_tensorcore(matA.astype(np.float16), matB.T.astype(np.float16), matC.copy().astype(np.float16), M, N, K)
        if not args.is_profile:
            tensorcore_abs_err, tensorcore_abs_rel, tensorcore_rel_err = abs_error_w_rel(matC_tensorcore_exp.copy(), matC_ref.astype(np.float16).copy())

        matC_cpasync_exp = matrix_mul_lib.mat_mul_cpasync(matA.astype(np.float16), matB.astype(np.float16), matC.copy().astype(np.float16), M, N, K)
        if not args.is_profile:
            cpasync_abs_err, cpasync_abs_rel, cpasync_rel_err = abs_error_w_rel(matC_cpasync_exp.copy(), matC_ref.astype(np.float16).copy())

        if not args.is_profile:
            print(f"naive version max abs err:\t\t\t{naive_abs_err} ({100*naive_abs_rel}%)")
            print(f"half version max abs err:\t\t\t{half_abs_err} ({100*half_abs_rel}%)")
            print(f"SIMT version max abs err:\t\t\t{simt_abs_err} ({100*simt_abs_rel}%)")
            print(f"SIMT with pipeline version max abs err:\t\t{simt_pipeline_abs_err} ({100*simt_pipeline_abs_rel}%)")
            print(f"tensorcore version max abs err:\t\t\t{tensorcore_abs_err} ({100*tensorcore_abs_rel}%)")
            print(f"cpasync version max abs err:\t\t\t{cpasync_abs_err} ({100*cpasync_abs_rel}%)")