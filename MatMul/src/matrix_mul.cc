#include "matrix_mul.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(matrix_mul_lib, m) {
    m.doc() = "Matrix Multiplication Library.";
    m.def("mat_mul_naive", &matrix_mul_naive_host, "The matrix multiplication function --- naive version.");
}