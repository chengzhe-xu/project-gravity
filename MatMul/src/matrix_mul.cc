#include "matrix_mul.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

PYBIND11_MODULE(matrix_mul_lib, m) {
    m.doc() = "Matrix Multiplication Library.";
    m.def("mat_mul_naive", &matrix_mul_naive_host, "The matrix multiplication function --- naive version.");
    m.def("mat_mul_half", &matrix_mul_half_host, "The matrix multiplication function --- half version.");
    m.def("mat_mul_simt", &matrix_mul_smit_host, "The matrix multiplication function --- SIMT version.");
    m.def("mat_mul_simt_pipeline", &matrix_mul_smit_pipeline_host, "The matrix multiplication function --- SIMT with pipeline version.");
    m.def("mat_mul_tensorcore", &matrix_mul_tensorcore_host, "The matrix multiplication function --- tensorcore version.");
    m.def("mat_mul_cpasync", &matrix_mul_cpasync_host, "The matrix multiplication function --- cp.async version.");
}