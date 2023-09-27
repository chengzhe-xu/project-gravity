cmake_minimum_required(VERSION 3.18)
project(matrix_mul_lib LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/pybind11 ./pybind11)

set(CMAKE_CXX_STANDARD 11)
set(PYBIND11_CPP_STANDARD -std=c++11)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/pybind11/include
    /usr/local/cuda/lib64
    /usr/local/cuda/include
)

include_directories(include)

add_library(matrix_mul_lib_kernel STATIC src/matrix_mul.cu)

set_target_properties(matrix_mul_lib_kernel PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    CUDA_SEPARABLE_COMPILATION ON
)

add_library(matrix_mul_lib MODULE src/matrix_mul.cc)

set_target_properties(matrix_mul_lib PROPERTIES 
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
target_link_libraries(matrix_mul_lib PRIVATE matrix_mul_lib_kernel)
target_link_libraries(matrix_mul_lib PRIVATE pybind11::module)
