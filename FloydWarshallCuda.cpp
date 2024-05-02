#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <chrono>
#include <iostream>
#include <iomanip>

namespace py = pybind11;

struct ResultWithRuntime {
    int* result;
    float runtime;
};

extern "C" {
    ResultWithRuntime run_blocked_fw_cuda(int** rand_adj_matrix, int number_of_vert);
    ResultWithRuntime run_naive_fw_cuda(int** rand_adj_matrix, int number_of_vert);
    int** generateRandomAdjacencyMatrix(int vertices, int density);
    void freeAdjacencyMatrix(int** adjacencyMatrix, int vertices);
    void floydWarshall_CPU(int vertices, int** graph);
}

// Wrapper function for Pybind11
std::tuple<py::array_t<int>, float> py_floydWarshall_CPU(py::array_t<int> input) {
    // Get buffer information
    py::buffer_info buf_info = input.request();


    // Cast the pointer
    int** graph = new int* [buf_info.shape[0]];
    for (int i = 0; i < buf_info.shape[0]; ++i) {
        graph[i] = static_cast<int*>(buf_info.ptr) + i * buf_info.shape[1];
    }

    auto start = std::chrono::steady_clock::now();
    // Call the original function
    floydWarshall_CPU(buf_info.shape[0], graph);

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    float durationFloat = static_cast<float>(duration.count());


    // Create a new NumPy array to return the modified graph
    auto result = py::array_t<int>({ buf_info.shape[0], buf_info.shape[1] });
    py::buffer_info result_buf_info = result.request();
    std::copy_n(graph[0], buf_info.size, static_cast<int*>(result_buf_info.ptr));

    // Free the memory allocated for graph
    delete[] graph;

    return std::make_tuple(result, durationFloat);
}

//Wrapper function for Pybind11
std::tuple<py::array_t<int>, float> py_run_blocked_fw_cuda(py::array_t<int> input) {
    // Extracting the 2D NumPy array from the Python argument
    py::buffer_info buf_info = input.request();

    // Allocate memory for an array of pointers to int
    int** rand_adj_matrix = new int* [buf_info.shape[0]];

    // Set the pointers to the beginning of each row
    for (int i = 0; i < buf_info.shape[0]; ++i) {
        rand_adj_matrix[i] = static_cast<int*>(buf_info.ptr) + i * buf_info.shape[1];
    }
    
    int number_of_vert = buf_info.shape[0];

    // Call the original function
    ResultWithRuntime result = run_blocked_fw_cuda(rand_adj_matrix, number_of_vert);
    delete[] rand_adj_matrix;

    // Create a new Pybind11 array to return the result
    py::array_t<int> output({ number_of_vert, number_of_vert });
    py::buffer_info result_buf_info = output.request();
    std::copy(result.result, result.result + number_of_vert * number_of_vert, static_cast<int*>(result_buf_info.ptr));

    // Don't forget to free the memory allocated in the original function
    delete[] result.result;

    return std::make_tuple(output, result.runtime);
}


//Wrapper function for Pybind11
std::tuple<py::array_t<int>, float> py_run_naive_fw_cuda(py::array_t<int> input) {
    // Extracting the 2D NumPy array from the Python argument
    py::buffer_info buf_info = input.request();

    // Allocate memory for an array of pointers to int
    int** rand_adj_matrix = new int* [buf_info.shape[0]];

    // Set the pointers to the beginning of each row
    for (int i = 0; i < buf_info.shape[0]; ++i) {
        rand_adj_matrix[i] = static_cast<int*>(buf_info.ptr) + i * buf_info.shape[1];
    }
    
    int number_of_vert = buf_info.shape[0];
    
    // Call the original function
    ResultWithRuntime result = run_naive_fw_cuda(rand_adj_matrix, number_of_vert);
    delete[] rand_adj_matrix;


    // Create a new Pybind11 array to return the result
    py::array_t<int> output({ number_of_vert, number_of_vert });
    py::buffer_info result_buf_info = output.request();
    std::copy(result.result, result.result + number_of_vert * number_of_vert, static_cast<int*>(result_buf_info.ptr));
   

    //free the memory allocated in the original function
    delete[] result.result;
   
    return std::make_tuple(output, result.runtime);
}

// Wrapper function for Pybind11
py::array_t<int> py_generateRandomAdjacencyMatrix(int vertices, int density) {
    // Generate the adjacency matrix
    int** adjacencyMatrix = generateRandomAdjacencyMatrix(vertices, density);

    // Convert the 2D array to a NumPy array
    py::array_t<int> result({ vertices, vertices });
    py::buffer_info buf_info = result.request();
    int* ptr = static_cast<int*>(buf_info.ptr);
    for (int i = 0; i < vertices; ++i) {
        for (int j = 0; j < vertices; ++j) {
            ptr[i * vertices + j] = adjacencyMatrix[i][j];
        }
    }

    // Free the memory allocated for the adjacency matrix
    freeAdjacencyMatrix(adjacencyMatrix, vertices);

    return result;
}


PYBIND11_MODULE(FloydWarshallCuda, m) {
    m.def("run_blocked_fw_cuda", &py_run_blocked_fw_cuda, "Run Blocked Floyd-Warshall on GPU");
    m.def("run_naive_fw_cuda", &py_run_naive_fw_cuda, "Run Naive Floyd-Warshall on GPU");
    m.def("generateRandomAdjacencyMatrix", &py_generateRandomAdjacencyMatrix, "Generate random adjacency matrix");
    m.def("floydWarshall_CPU", &py_floydWarshall_CPU, "Run Floyd-Warshall algorithm on CPU");
}
