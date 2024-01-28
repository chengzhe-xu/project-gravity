cmake_minimum_required(VERSION 3.16)
project(mat_trans_lib LANGUAGES CXX CUDA)

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

# atomAdd version
add_library(mat_trans_lib_naive STATIC src/mat_trans_naive.cu)
set_target_properties(mat_trans_lib_naive PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 87
)

add_library(mat_trans_lib MODULE src/mat_trans.cc)

set_target_properties(mat_trans_lib PROPERTIES 
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    CUDA_ARCHITECTURES 87
)
target_link_libraries(mat_trans_lib PRIVATE mat_trans_lib_naive)
target_link_libraries(mat_trans_lib PRIVATE pybind11::module)
