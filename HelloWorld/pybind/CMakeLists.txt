cmake_minimum_required(VERSION 3.14)

project(hello_world_cpu_pybind)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/pybind11 ./pybind11)
pybind11_add_module(hello_world_cpu_pybind src/hello_world_cpu.cc)
