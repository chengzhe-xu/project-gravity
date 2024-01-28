import numpy as np
import argparse
import build.mat_trans_lib as mat_trans_lib

def abs_error_w_rel(mat_2d, mat_ref_2d, esp=1e-9):
    mat = mat_2d.reshape([1, -1])[0]
    mat_ref = mat_ref_2d.reshape([1, -1])[0]
    abs_err = np.max(np.abs(mat - mat_ref))
    idx = np.argwhere(np.abs(mat - mat_ref) == abs_err).reshape([1, -1])[0]
    abs_rel = np.max(np.abs((mat[idx] - mat_ref[idx]) / (mat_ref[idx] + esp)))
    rel_err = np.max(np.abs((mat - mat_ref) / (mat_ref + esp)))
    return abs_err, abs_rel, rel_err

def argument_parser():
    parser = argparse.ArgumentParser(description="Matrix transpose arg parser.")
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--N", type=int, default=2048)
    parser.add_argument("--test_round", type=int, default=5)
    return parser.parse_args()

if __name__=='__main__':
    args = argument_parser()
    M, N = args.M, args.N
    np.random.seed(np.random.randint(7))
    for _ in range(args.test_round):
        mat = (np.random.rand(M, N) - 0.5) * 2 * 10
        matT = (np.random.rand(M, N) - 0.5) * 2 * 10
        matT_ref = mat.T.copy()

        matT_naive = mat_trans_lib.mat_trans_naive(mat, matT, M, N)
        naive_abs_err, naive_abs_rel, _ = abs_error_w_rel(matT_naive.copy(), mat.T.copy())
        
        print(f"naive version matrix transpose abs err:\t\t\t{naive_abs_err} ({100*naive_abs_rel}%)")