cmake_minimum_required(VERSION 3.14)

project(hello_world_cuda)

set(CMAKE_CXX_STANDARD 11)

FIND_PACKAGE(CUDA REQUIRED)

CUDA_ADD_EXECUTABLE(hello_world_cuda src/hello_world_cuda.cu)
TARGET_LINK_LIBRARIES(hello_world_cuda)
