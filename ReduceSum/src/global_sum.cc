#include "global_sum.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

PYBIND11_MODULE(global_sum_lib, m) {
    m.doc() = "Global Sum Library.";
    m.def("global_sum_atomAdd", &global_sum_atomAdd_host, "The global sum function --- naive atomAdd version.");
    m.def("global_sum_reduce", &global_sum_reduce_host, "The global sum function --- reduce sum version.");
    m.def("global_sum_reduce_async", &global_sum_reduce_async_host, "The global sum function --- reduce sum version.");
}