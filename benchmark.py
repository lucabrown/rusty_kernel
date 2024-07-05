from collections import defaultdict
import os
from re import T
from networkx import k_truss, powerlaw_cluster_graph
import numpy as np
import time
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram, EdgeHistogram, NeighborhoodHash, WeisfeilerLehmanOptimalAssignment
import scipy as sp
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

f = './DATA'

random_state = 42


graph_kernels = [
    NeighborhoodHash(normalize=True, random_state=random_state, R=1, nh_type='count_sensitive'),
]

dict = {}

for kernel in graph_kernels:
    print(f"Training {kernel}")

    results = []

    for folder in os.listdir(f):
        gk = rusty_kernel.PyGraphKernel() # type: ignore

        print(f"- On {folder}")

        folder_path = f"{f}/{folder}/{folder}"

        start_time = time.time()

        graphs, labels = read_data(folder_path)

        folder_read_time = time.time()

        G_train, G_test, y_train, y_test = train_test_split(graphs, labels, test_size=0.1, random_state=random_state)
 
        
        data_fit_time = time.time()
        
        K_train_p = kernel.fit_transform(G_train)
        K_test_p = kernel.transform(G_test)

        p_train_time = time.time()
        
        K_train_r = gk.fit_transform(transform_data(G_train))
        K_test_r = gk.transform(transform_data(G_test)).T

        end_time = time.time()

        print("Read time: {:.3f} s".format(folder_read_time - start_time))


        print("Python fit time: {:.3f} s".format(p_train_time - data_fit_time))
        print("Rust fit time:   {:.3f} s".format(end_time - p_train_time))
        print()
        
        clf_p = SVC(kernel='precomputed')
        clf_p.fit(K_train_p, y_train)
        y_pred_p = clf_p.predict(K_test_p)

        clf_r = SVC(kernel='precomputed')
        clf_r.fit(K_train_r, y_train)
        y_pred_r = clf_r.predict(K_test_r)

        accuracy_p = accuracy_score(y_test, y_pred_p)
        accuracy_r = accuracy_score(y_test, y_pred_r)
        print(f"Python accuracy: {accuracy_p * 100:.2f} %")
        print(f"Rust accuracy:   {accuracy_r * 100:.2f} %\n")

        
    dict[kernel] = results