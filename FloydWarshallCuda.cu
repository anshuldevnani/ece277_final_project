#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <chrono>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#define INF 9999999
#define BLOCK_DIM 16

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




__global__ void floydWarshall(int* graph, int vertices, int k) {
    // Calculate the row index for the current thread within the grid
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column index for the current thread within the grid
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread's indices are within the valid range of vertices
    if (i < vertices && j < vertices) {
        // Calculate the flattened index corresponding to the 2D indices (i, j)
        int index = i * vertices + j;

        // Calculate the sum of distances from vertex i to k and from k to j
        int ikj = graph[i * vertices + k] + graph[k * vertices + j];

        // Update the graph matrix at index (i, j) if the computed path is shorter
        graph[index] = (graph[index] > ikj) ? ikj : graph[index];
    }
}

void printSolution(int* dist, int V) {
    std::cout << "Shortest distances between every pair of vertices GPU:\n";
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (dist[i * V + j] == INF)
                std::cout << "INF\t";
            else
                std::cout << dist[i * V + j] << "\t";
        }
        std::cout << "\n";
    }
}


int** generateRandomAdjacencyMatrix(int vertices, int density) {
    // Allocate memory for a 2D array on the heap
    int** adjacencyMatrix = new int* [vertices];
    for (int i = 0; i < vertices; ++i) {
        adjacencyMatrix[i] = new int[vertices];
    }


    // Initialize random seed
    std::srand(std::time(nullptr));

    // Initialize the adjacency matrix with random weights based on density
    for (int i = 0; i < vertices; ++i) {
        for (int j = 0; j < vertices; ++j) {
            if (i == j) {
                adjacencyMatrix[i][j] = 0;  // No self-loops
            }
            else {
                if (std::rand() % 100 < density) {
                    adjacencyMatrix[i][j] = std::rand() % 10 + 1;  // Adjust weight range as needed
                }
                else {
                    adjacencyMatrix[i][j] = INF;  // No edge
                }
            }
        }
    }

    return adjacencyMatrix;
}

// Function to free the memory allocated for the adjacency matrix
void freeAdjacencyMatrix(int** adjacencyMatrix, int vertices) {
    for (int i = 0; i < vertices; ++i) {
        delete[] adjacencyMatrix[i];
    }
    delete[] adjacencyMatrix;
}

void printMatrix_2(int** matrix, int vertices) {
    for (int i = 0; i < vertices; ++i) {
        for (int j = 0; j < vertices; ++j) {
            if (matrix[i][j] == INF)
            {
                std::cout << "INF" << "\t";
            }
            else
            {
                std::cout << matrix[i][j] << "\t";
            }
            
        }
        std::cout << "\n";
    }
}

void freeMatrix(int** matrix, int vertices) {
    for (int i = 0; i < vertices; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

void floydWarshall_CPU(int vertices, int ** graph) {
    for (int k = 0; k < vertices; k++) {
        for (int i = 0; i < vertices; i++) {
            for (int j = 0; j < vertices; j++) {
                if (graph[i][k] != INF && graph[k][j] != INF &&
                    graph[i][k] + graph[k][j] < graph[i][j]) {
                    graph[i][j] = graph[i][k] + graph[k][j];
                }
            }
        }
    }
}


__host__ void flattenAndAllocateOnGPU(int** h_matrix, int** d_matrix, int vertices) {

    // Flatten the 2D array into a 1D array
    int* flatMatrix = new int[vertices * vertices];

    for (int i = 0; i < vertices; ++i) {
        for (int j = 0; j < vertices; ++j) {
            flatMatrix[i * vertices + j] = h_matrix[i][j];
        }
    }
    // Allocate memory on the GPU
    cudaMalloc((void**)d_matrix, vertices * vertices * sizeof(int));

    // Copy the flattened matrix to the GPU
    cudaMemcpy(*d_matrix, flatMatrix, vertices * vertices * sizeof(int), cudaMemcpyHostToDevice);

    // Free the temporary flat array on the host
    delete[] flatMatrix;
}

// Function to check if two matrices are equal
void areMatricesEqual(int** matrix1, int* matrix2, int vertices) {
    for (int i = 0; i < vertices; ++i) {
        for (int j = 0; j < vertices; ++j) {
            if (matrix1[i][j] != matrix2[i * vertices + j]) {
                printf("Failure: Matrices are not identical.\n");
                return;
            }
        }
    }
    printf("Success: Matrices are identical.\n");
}


__global__ void fw_phase_1_kernel(int* graph, int n, int k )
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate offset for the current iteration
    int offset = (k * BLOCK_DIM);
    
    // Load data from global memory to shared memory
    __shared__ int shared_graph[BLOCK_DIM][BLOCK_DIM];

    // sync
    __syncthreads();

    shared_graph[ty][tx] = graph[(offset + ty) * n + (offset + tx)];

    // perform sync
    __syncthreads();
    
    // Perform the Floyd-Warshall computation
    int sum = 0;
    for (int k = 0; k < BLOCK_DIM; ++k) {
        sum = shared_graph[ty][k] + shared_graph[k][tx];

        // Update the minimum distance in the shared memory
        if (sum < shared_graph[ty][tx]) {
            shared_graph[ty][tx] = sum;
        }
    }
    __syncthreads();

    graph[(offset + ty) * n + (offset + tx)] = shared_graph[ty][tx];
}


__global__ void fw_phase_2_kernel(int* graph, int n, int k) {
    // Thread indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int offset = k * BLOCK_DIM;

    // Check if the block index is not equal to the current stage
    if (bx != k) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        // Calculate indices for the primary and temporary matrices
        int i_prime = offset + ty;
        int j_prime = offset + tx;

        // Determine indices for the current thread's matrix elements
        int i = (by != 0) ? BLOCK_DIM * bx + ty : i_prime;
        int j = (by != 0) ? j_prime : BLOCK_DIM * bx + tx;

        // Shared memory for the current thread's matrix and the temporary matrix
        __shared__ int local_matrix[BLOCK_DIM][BLOCK_DIM];
        __shared__ int temp_matrix[BLOCK_DIM][BLOCK_DIM];

        // Load data into shared memory
        local_matrix[ty][tx] = graph[i * n + j];
        temp_matrix[ty][tx] = graph[i_prime * n + j_prime];

        // Synchronize threads to ensure all data is loaded into shared memory
        __syncthreads();

        int sum = 0;
        // Perform matrix addition for the specific block
        if (by != 0) {
            for (int k = 0; k < BLOCK_DIM; ++k) {
                sum = local_matrix[ty][k] + temp_matrix[k][tx];
                // Update the local matrix element if a smaller sum is found
                local_matrix[ty][tx] = (sum < local_matrix[ty][tx]) ? sum : local_matrix[ty][tx];
            }
        }
        else {
            for (int k = 0; k < BLOCK_DIM; ++k) {
                sum = temp_matrix[ty][k] + local_matrix[k][tx];
                // Update the local matrix element if a smaller sum is found
                local_matrix[ty][tx] = (sum < local_matrix[ty][tx]) ? sum : local_matrix[ty][tx];
            }
        }

        // Synchronize threads to ensure all data is loaded into local matrix
        __syncthreads();

        // Store the result back to global memory
        graph[i * n + j] = local_matrix[ty][tx];
    }
}

__global__ void fw_phase_3_kernel(int* graph, int n, int k) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.y;
    int by = blockIdx.x;

    // Check if both block indices are different from the current stage
    if (bx != k && by != k) {

        int offset = k * BLOCK_DIM;
        // Calculate indices for the matrix elements
        int i = BLOCK_DIM * by + ty;
        int j = BLOCK_DIM * bx + tx;

        int i_row = offset + ty;
        int j_col = offset + tx;

        // Shared memory for the row and column matrices
        __shared__ int row_matrix[BLOCK_DIM][BLOCK_DIM];
        __shared__ int col_matrix[BLOCK_DIM][BLOCK_DIM];

        // Load data into shared memory
        int graph_ij = graph[i * n + j];
        row_matrix[ty][tx] = graph[i_row * n + j];
        col_matrix[ty][tx] = graph[i * n + j_col];

        // Synchronize threads to ensure all data is loaded into shared memory
        __syncthreads();

        int s = 0;
        // Perform matrix addition for the specific block
        for (int k = 0; k < BLOCK_DIM; ++k) {
            s = col_matrix[ty][k] + row_matrix[k][tx];
            // Update graph_ij if a smaller sum is found
            graph_ij = (s < graph_ij) ? s : graph_ij;
        }

        // Synchronize threads again
        __syncthreads();

        // Store the result back to global memory
        graph[i * n + j] = graph_ij;
    }
}

ResultWithRuntime run_naive_fw_cuda(int ** rand_adj_matrix, int number_of_vert)
{

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int* d_matrix;

    flattenAndAllocateOnGPU(rand_adj_matrix, &d_matrix, number_of_vert);


    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((number_of_vert + threadsPerBlock.x - 1) / threadsPerBlock.x, (number_of_vert + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEventRecord(start);
    for (int k = 0; k < number_of_vert; k++) {
        floydWarshall << <numBlocks, threadsPerBlock >> > (d_matrix, number_of_vert, k);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
   
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    int* h_result = new int[number_of_vert * number_of_vert];
    cudaMemcpy(h_result, d_matrix, number_of_vert * number_of_vert * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);
    return { h_result, milliseconds };
}

ResultWithRuntime run_blocked_fw_cuda(int** rand_adj_matrix, int number_of_vert)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int* d_matrix;
    flattenAndAllocateOnGPU(rand_adj_matrix, &d_matrix, number_of_vert);

    const int blocks = (number_of_vert + BLOCK_DIM - 1) / BLOCK_DIM;
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 phase3_grid(blocks, blocks, 1);


    cudaEventRecord(start);
    for (int k = 0; k < blocks; k++) {

        fw_phase_1_kernel << <1, block_dim >> > (d_matrix, number_of_vert, k);

        fw_phase_2_kernel << <blocks, block_dim >> > (d_matrix, number_of_vert, k);

        fw_phase_3_kernel << <phase3_grid, block_dim >> > (d_matrix, number_of_vert, k);

    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
  
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    int* h_result = new int[number_of_vert * number_of_vert];
    cudaMemcpy(h_result, d_matrix, number_of_vert * number_of_vert * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);
    return { h_result, milliseconds };

}