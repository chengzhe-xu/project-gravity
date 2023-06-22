#include "hello_world_cuda.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(hello_world_cuda_pybind, m) {
    m.doc() = "Hello World from CUDA (pybind11)";
    m.def("hello_world", &hello_world_cuda_host, "The hello world function.");
}