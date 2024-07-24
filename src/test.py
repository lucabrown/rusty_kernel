import os
import numpy as np
import time
from grakel import Graph
from grakel.kernels import NeighborhoodHash
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import rusty_kernel

GREEN = "\033[92m"
RESET = "\033[0m"

def read_data(folder_path):
    # Read the adjacency matrix file
    edges = []
    with open(f"{folder_path}_A.txt", "r") as f:
        for line in f:
            edges.append(tuple(map(int, line.strip().split(','))))
    
    # Read the graph indicator file
    graph_indicator = []
    with open(f"{folder_path}_graph_indicator.txt", "r") as f:
        for line in f:
            graph_indicator.append(int(line.strip()))
    
    # Read the node labels file
    node_labels = []
    with open(f"{folder_path}_node_labels.txt", "r") as f:
        for line in f:
            node_labels.append(int(line.strip()))
    
    # Read the graph labels file (targets)
    graph_labels = []
    with open(f"{folder_path}_graph_labels.txt", "r") as f:
        for line in f:
            graph_labels.append(int(line.strip()))

    # Create a dictionary to store the graph information
    graph_dict = {}
    for (node1, node2) in edges:
        graph_id1 = graph_indicator[node1 - 1]
        graph_id2 = graph_indicator[node2 - 1]
        if graph_id1 == graph_id2:
            if graph_id1 not in graph_dict:
                graph_dict[graph_id1] = {'edges': [], 'nodes': set()}
            graph_dict[graph_id1]['edges'].append((node1 - 1, node2 - 1))
            graph_dict[graph_id1]['nodes'].update([node1 - 1, node2 - 1])
    
    # Create adjacency matrices and node labels for each graph
    graphs = []
    labels = []
    for graph_id, graph_info in graph_dict.items():
        nodes = sorted(list(graph_info['nodes']))
        node_map = {node: idx for idx, node in enumerate(nodes)}
        size = len(nodes)
        
        adjacency_matrix = np.zeros((size, size), dtype=int)
        for (node1, node2) in graph_info['edges']:
            adjacency_matrix[node_map[node1], node_map[node2]] = 1
            adjacency_matrix[node_map[node2], node_map[node1]] = 1
        
        node_label_dict = {node_map[node]: node_labels[node] for node in nodes}
        
        # Initialize the graph with adjacency matrix and node labels
        graph = Graph(initialization_object=adjacency_matrix, node_labels=node_label_dict)
        graphs.append(graph)
        labels.append(graph_labels[graph_id - 1])  # Graph IDs are 1-based, so subtract 1
    
    return graphs, labels

def transform_data(graphs):
    list = []

    for graph in graphs:
        list.append((graph.adjacency_matrix, graph.index_node_labels, len(graph.adjacency_matrix)))

    return list

f = './TEST'

random_state = 42

n = 10

results = []

for folder in os.listdir(f):

    print(f"- On {folder}")

    folder_path = f"{f}/{folder}/{folder}"

    start_time = time.time()

    graphs, labels = read_data(folder_path)

    folder_read_time = time.time()

    csnh_average_time = 0
    wnh_average_time = 0

    csnh_average_accuracy = 0
    wnh_average_accuracy = 0

    csnh_values = []
    wnh_values = []

    for i in range(n):
        G_train, G_test, y_train, y_test = train_test_split(graphs, labels, test_size=0.1)

        csnh = rusty_kernel.PyGraphKernel(0) # type: ignore
        wnh = rusty_kernel.PyGraphKernel(1) # type: ignore

        data_fit_time = time.time()

        K_train_wnh = wnh.fit_transform(transform_data(G_train))
        K_test_wnh = wnh.transform(transform_data(G_test)).T

        wnh_train_time = time.time()

        K_train_csnh = csnh.fit_transform(transform_data(G_train))
        K_test_csnh = csnh.transform(transform_data(G_test)).T

        end_time = time.time()

        wnh_average_time += wnh_train_time - data_fit_time
        csnh_average_time += end_time - wnh_train_time

        clf_wnh = SVC(kernel='precomputed')
        clf_wnh.fit(K_train_wnh, y_train)
        y_pred_wnh = clf_wnh.predict(K_test_wnh)

        clf_csnh = SVC(kernel='precomputed')
        clf_csnh.fit(K_train_csnh, y_train)
        y_pred_csnh = clf_csnh.predict(K_test_csnh)

        accuracy_wnh = accuracy_score(y_test, y_pred_wnh)
        accuracy_r = accuracy_score(y_test, y_pred_csnh)

        wnh_values.append(accuracy_wnh)
        csnh_values.append(accuracy_r)

        print("Percentage done: {:.2f}%".format((i + 1) / n * 100), end="\r")


    wnh_average_time /= n
    csnh_average_time /= n

    wnh_average_accuracy = sum(wnh_values) / n
    csnh_average_accuracy = sum(csnh_values) / n

    wnh_standard_deviation = np.std(wnh_values)
    csnh_standard_deviation = np.std(csnh_values)

    print("Read time: {:.3f} s".format(folder_read_time - start_time))

    print("WNH fit time:    {:.3f} s".format(wnh_average_time))
    print("CSNH fit time:   {:.3f} s".format(csnh_average_time))
    print()

    # print accuracy and standard deviation
    print(f"WNH accuracy:    {wnh_average_accuracy * 100:.2f} % ± {wnh_standard_deviation * 100:.2f}")
    print(f"CSNH accuracy:   {csnh_average_accuracy * 100:.2f} % ± {csnh_standard_deviation * 100:.2f}\n")