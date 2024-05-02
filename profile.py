import sys
sys.path.append('./build/Release')
import FloydWarshallCuda
import numpy as np
import gc
import argparse

# function to print profile table
def print_ascii_table(header, data):
    # Calculate the column widths
    column_widths = [max(len(str(item)) for item in col) for col in zip(header, *data)]

    # Print the header row
    print("|".join(f"{col:<{width}}" for col, width in zip(header, column_widths)))
    print("|".join("-" * width for width in column_widths))

    # Print the data rows
    for row in data:
        print("|".join(f"{col:<{width}}" for col, width in zip(row, column_widths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vertices-list', type=str, help='List of vertices separated by commas', required=True)
    args = parser.parse_args()

    # Table variables
    header = ["Number of Vertices", "FW CPU Runtime (ms)", "Speedup [Runtime (ms)] Naive FW GPU", "Speedup [Runtime (ms)] Blocked FW GPU", "All Resulting Matrices Match?"]
    data = []

    # read in from command line
    vertices_list = [int(vertex) for vertex in args.vertices_list.split(',')]


    # Loop over the number of vertices 
    for i in (vertices_list):
        sub_data_list = []
        sub_data_list.append(str(i))

        # generate random adj matrix with i vertices an 50 % probs that nodes will be connected
        # with a random value weighted edge 
        random_adj_matrix = FloydWarshallCuda.generateRandomAdjacencyMatrix(i, 50)

        # run CPU FW and capture resulting matrix and runtime
        cpu_fw_truth_matrix, cpu_runtime = (FloydWarshallCuda.floydWarshall_CPU(random_adj_matrix))
        sub_data_list.append(str(round(cpu_runtime,2)))

        # run naive and blocked FW and capture resulting matrix and 
        gpu_fw_naive_final_matrix, gpu_fw_naive_runtime = (FloydWarshallCuda.run_naive_fw_cuda(random_adj_matrix))
        gpu_fw_blocked_final_matrix, gpu_fw_blocked_runtime = (FloydWarshallCuda.run_blocked_fw_cuda(random_adj_matrix))

        # Calc speedup
        speed_up_naive = cpu_runtime / gpu_fw_naive_runtime
        speed_up_blocked = cpu_runtime / gpu_fw_blocked_runtime

        sub_data_list.append(str(round(speed_up_naive, 2)) + " [ " + str(round(gpu_fw_naive_runtime,2)) + " ]" )
        sub_data_list.append(str(round(speed_up_blocked,2)) + " [ " + str(round(gpu_fw_blocked_runtime,2)) + " ]" )

        # Check if all matrices match
        result = all(np.array_equal(arr, cpu_fw_truth_matrix) for arr in [gpu_fw_naive_final_matrix, gpu_fw_blocked_final_matrix])
        if(result):
            sub_data_list.append("True")
        else:
            sub_data_list.append("False")

        data.append(sub_data_list)

        # mem managment
        del random_adj_matrix
        del cpu_fw_truth_matrix
        del gpu_fw_naive_final_matrix
        del gpu_fw_blocked_final_matrix
        gc.collect()

    # print results
    print_ascii_table(header, data)


if __name__ == "__main__":
    main()