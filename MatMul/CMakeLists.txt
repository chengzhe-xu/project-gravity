cmake_minimum_required(VERSION 3.16)
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

# common helper functions
add_library(matrix_mul_lib_common_helpers STATIC src/common.cu)
set_target_properties(matrix_mul_lib_common_helpers PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 87
)
# naive version
add_library(matrix_mul_lib_naive STATIC src/matrix_mul_naive.cu)
set_target_properties(matrix_mul_lib_naive PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 87
)
target_link_libraries(matrix_mul_lib_naive PRIVATE matrix_mul_lib_common_helpers)
# half version
add_library(matrix_mul_lib_half STATIC src/matrix_mul_half.cu)
set_target_properties(matrix_mul_lib_half PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 87
)
target_link_libraries(matrix_mul_lib_half PRIVATE matrix_mul_lib_common_helpers)
# SIMT version
add_library(matrix_mul_lib_simt STATIC src/matrix_mul_simt.cu)
set_target_properties(matrix_mul_lib_simt PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 87
)
target_link_libraries(matrix_mul_lib_simt PRIVATE matrix_mul_lib_common_helpers)

add_library(matrix_mul_lib MODULE src/matrix_mul.cc)

set_target_properties(matrix_mul_lib PROPERTIES 
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    CUDA_ARCHITECTURES 87
)
target_link_libraries(matrix_mul_lib PRIVATE matrix_mul_lib_naive)
target_link_libraries(matrix_mul_lib PRIVATE matrix_mul_lib_half)
target_link_libraries(matrix_mul_lib PRIVATE matrix_mul_lib_simt)
target_link_libraries(matrix_mul_lib PRIVATE pybind11::module)
