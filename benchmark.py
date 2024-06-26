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
# random_state = random.randint(1, 1000)

# graphs, labels = read_data("./DATA/MUTAG/MUTAG")

# for graph in graphs:
#     print(graph.adjacency_matrix)
#     print(graph.index_node_labels)

graph_kernels = [
    # WeisfeilerLehman(n_iter=30, normalize=True),
    # VertexHistogram(normalize=True),
    # EdgeHistogram(normalize=True),
    NeighborhoodHash(normalize=True, random_state=random_state, R=1, nh_type='count_sensitive'),
    # WeisfeilerLehmanOptimalAssignment(n_iter=5, normalize=True)
]

dict = {}

for kernel in graph_kernels:
    print(f"Training {kernel}")

    # m = rusty_kernel.transform(graphs)


    
    # print("Kernel m size: ", m)

    results = []

    for folder in os.listdir(f):
        gk = rusty_kernel.PyGraphKernel()

        print(f"- On {folder}")

        folder_path = f"{f}/{folder}/{folder}"

        start_time = time.time()

        graphs, labels = read_data(folder_path)

        folder_read_time = time.time()

        G_train, G_test, y_train, y_test = train_test_split(graphs, labels, test_size=0.1, random_state=random_state)
        
        # data = gk.fit_transform(transform_data(G_train))

        # print("Data: ", data)  
        # print("Data shape: ", data.shape) 

        # print("Graphs type: ", type(G_train))
        # print("Type of G_train[0]: ", type(G_train[0]))
        # print("Graphs[0] type: ", type(G_train[0].adjacency_matrix))
        # print("Graphs[0] type: ", type(G_train[0].index_node_labels))
        # matrix = kernel.fit_transform(graphs)
        
        data_fit_time = time.time()
        
        # K_train = kernel.fit_transform(G_train)
        K_train = gk.fit_transform(transform_data(G_train))
        # print K_train shape
        # print("K_train shape: ", K_train.shape)
        # print("K_train shape: ", K_train)

        # K_test = kernel.transform(G_test)
        K_test = gk.transform(transform_data(G_test))

        # print("K_test type: ", type(K_test))
        # print("K_test shape: ", K_test.shape)

        # K_test = K_test.T

        # print("K_test type: ", type(K_test))
        # print("K_test shape: ", K_test.shape)
        # print("K_test shape: ", K_test)

        end_time = time.time()

        # print(f"- {end_time - start_time:.2f} s")
        print("Read time: {:.3f} s".format(folder_read_time - start_time))

        print("Fit time: {:.3f} s".format(end_time - folder_read_time))
        print()
        # print("Fitted dataset: ", kernel.X[0][0])

        # print kernel matrix size
        # print("Kernel matrix size: ", matrix.shape)
        # print("Kernel matrix size: ", matrix)

        # // count how many values are 0 in the matrix out od the total
        # print("Sparsity: ", (matrix == 0).sum() / matrix.size)
        
        # clf = SVC(kernel='precomputed')
        # clf.fit(K_train, y_train)
        # y_pred = clf.predict(K_test)

        # accuracy = accuracy_score(y_test, y_pred)
        # print(f"- {accuracy * 100:.2f} %")

        # results.append((end_time - start_time, accuracy, folder))
        # break

    dict[kernel] = results



# results_by_dataset = defaultdict(list)
# for kernel, results in dict.items():
#     for time, accuracy, dataset in results:
#         results_by_dataset[dataset].append((kernel, time, accuracy))

# # Calculate averages and determine highest accuracy per dataset
# for dataset, results in results_by_dataset.items():
#     print(f"~~~~~~~~~~~~~~~~ {dataset} ~~~~~~~~~~~~~~~~")
    
#     # Calculate averages
#     averages = []
#     for kernel, time, accuracy in results:
#         av_time = sum(time for k, time, acc in results if k == kernel) / len([k for k, t, a in results if k == kernel])
#         av_accuracy = sum(accuracy for k, t, acc in results if k == kernel) / len([k for k, t, a in results if k == kernel])
#         averages.append((kernel, av_time, av_accuracy))
    
#     # Find the highest accuracy
#     highest_accuracy = max(averages, key=lambda x: x[2])
    
#     # Print results with highest accuracy in bold
#     for kernel, av_time, av_accuracy in averages:
#         accuracy_str = f"{av_accuracy * 100:.2f} %"
#         if kernel == highest_accuracy[0]:
#             accuracy_str = f"{GREEN}{accuracy_str}{RESET}"
#         print(f"~ {kernel}:")
#         print(f"~ Average time: {av_time:.2f} s")
#         print(f"~ Average accuracy: {accuracy_str}")

#         print()

#         print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

