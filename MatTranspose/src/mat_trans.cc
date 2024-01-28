#include "mat_trans.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

PYBIND11_MODULE(mat_trans_lib, m) {
    m.doc() = "Matrix Transpose Library.";
    m.def("mat_trans_naive", &mat_trans_naive_host, "The matrix transpose function --- naive version.");
    m.def("mat_trans_shared", &mat_trans_shared_host, "The matrix transpose function --- shared memory version.");
}