cmake_minimum_required(VERSION 3.12)
project(FloydWarshallCuda)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)


find_package(CUDA REQUIRED)
find_package(PythonInterp 3.0 REQUIRED)
find_package(PythonLibs 3.0 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/pybind11/include)

find_package(CUDA REQUIRED)

cuda_add_library(FloydWarshallCuda SHARED FloydWarshallCuda.cpp FloydWarshallCuda.cu)

target_link_libraries(FloydWarshallCuda ${PYTHON_LIBRARIES} ${CUDA_LIBRARIES})

set_target_properties(FloydWarshallCuda PROPERTIES PREFIX "")