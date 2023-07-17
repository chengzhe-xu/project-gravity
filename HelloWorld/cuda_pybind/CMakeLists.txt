cmake_minimum_required(VERSION 3.18)
project(hello_world_cuda_pybind LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/pybind11 ./pybind11)

set(CMAKE_CXX_STANDARD 11)
set(PYBIND11_CPP_STANDARD -std=c++11)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/pybind11/include
    /usr/local/cuda/lib64
    /usr/local/cuda/include
)

add_library(hello_world_cuda_pybind_kernel STATIC src/hello_world_cuda.cu)

set_target_properties(hello_world_cuda_pybind_kernel PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    CUDA_SEPARABLE_COMPILATION ON
)

add_library(hello_world_cuda_pybind MODULE src/hello_world_cuda_pybind.cc)

set_target_properties(hello_world_cuda_pybind PROPERTIES 
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

target_link_libraries(hello_world_cuda_pybind PRIVATE hello_world_cuda_pybind_kernel)
target_link_libraries(hello_world_cuda_pybind PRIVATE pybind11::module)
