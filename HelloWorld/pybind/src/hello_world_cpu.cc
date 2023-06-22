#include <stdio.h>
#include <pybind11/pybind11.h>

int main() {
    printf("Hello World.\n");
    return 0;
}

PYBIND11_MODULE(hello_world_cpu_pybind, m) {
    m.doc() = "Hello World (pybind11)";
    m.def("hello_world", &main, "The hello world function.");
}