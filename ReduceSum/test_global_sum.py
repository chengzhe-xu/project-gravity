import numpy as np
import argparse
import build.global_sum_lib as global_sum_lib

def argument_parser():
    parser = argparse.ArgumentParser(description="Global sum arg parser.")
    parser.add_argument("--arr_size", type=int, default=4096)
    parser.add_argument("--test_round", type=int, default=5)
    return parser.parse_args()

if __name__=='__main__':
    args = argument_parser()
    arr_size = args.arr_size
    np.random.seed(np.random.randint(7))
    for _ in range(args.test_round):
        arr = (np.random.rand(1, arr_size) - 0.5) * 2 * 10
        arr = np.ascontiguousarray(arr.astype(np.float32))
        ref_sum_result = np.sum(arr)

        sum_result_atomAdd = global_sum_lib.global_sum_atomAdd(arr, arr_size)
        
        print(f"refrence version:\t{ref_sum_result}, atomAdd version:\t{sum_result_atomAdd}, abs error:\t{sum_result_atomAdd - ref_sum_result}")