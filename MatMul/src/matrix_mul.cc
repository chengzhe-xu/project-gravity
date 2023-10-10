#include "matrix_mul.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

PYBIND11_MODULE(matrix_mul_lib, m) {
    m.doc() = "Matrix Multiplication Library.";
    m.def("mat_mul_naive", &matrix_mul_naive_host, "The matrix multiplication function --- naive version.");
    m.def("mat_mul_simt", &matrix_mul_smit_host, "The matrix multiplication function --- SIMT version.");
}