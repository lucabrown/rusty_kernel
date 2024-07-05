mod graph;
mod graph_kernel;
mod neighbourhood_hash_kernel;
mod wasserstein_hash_kernel;

use graph::Graph;
use graph_kernel::GraphKernel;
use neighbourhood_hash_kernel::NeighbourhoodHashKernel;
use wasserstein_hash_kernel::WassersteinHashKernel;

use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rustc_hash::FxHashMap;
use std::iter::FromIterator;

use std::{
    collections::HashSet,
    fs::{self, File},
    io::{self, BufRead},
};

// TODO: change vertices to be index and node_dict to be index
fn read_data(folder_path: &str) -> io::Result<(Vec<Graph>, Vec<i32>)> {
    // Read the adjacency matrix file
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let edge_file_path: String = format!("{}_A.txt", folder_path);
    let edge_file: File = File::open(edge_file_path)?;
    for line in io::BufReader::new(edge_file).lines() {
        if let Ok(l) = line {
            let nodes: Vec<usize> = l.split(',').map(|s| s.trim().parse().unwrap()).collect();
            edges.push((nodes[0], nodes[1]));
        }
    }

    // Read the graph indicator file
    let mut graph_indicator: Vec<usize> = Vec::new();
    let graph_indicator_file_path: String = format!("{}_graph_indicator.txt", folder_path);
    let graph_indicator_file: File = File::open(graph_indicator_file_path)?;
    for line in io::BufReader::new(graph_indicator_file).lines() {
        if let Ok(l) = line {
            graph_indicator.push(l.trim().parse().unwrap());
        }
    }

    // Read the node labels file
    let mut node_labels: Vec<i32> = Vec::new();
    let node_labels_file_path: String = format!("{}_node_labels.txt", folder_path);
    let node_labels_file: File = File::open(node_labels_file_path)?;
    for line in io::BufReader::new(node_labels_file).lines() {
        if let Ok(l) = line {
            node_labels.push(l.trim().parse().unwrap());
        }
    }

    // Read the graph labels file (targets)
    let mut graph_labels: Vec<i32> = Vec::new();
    let graph_labels_file_path: String = format!("{}_graph_labels.txt", folder_path);
    let graph_labels_file: File = File::open(graph_labels_file_path)?;
    for line in io::BufReader::new(graph_labels_file).lines() {
        if let Ok(l) = line {
            graph_labels.push(l.trim().parse().unwrap());
        }
    }

    // Create a dictionary to store the graph information
    let mut graph_dict: FxHashMap<usize, (Vec<(usize, usize)>, HashSet<usize>)> =
        FxHashMap::default();
    for (node1, node2) in edges {
        let graph_id1: usize = graph_indicator[node1 - 1];
        let graph_id2: usize = graph_indicator[node2 - 1];
        if graph_id1 == graph_id2 {
            graph_dict
                .entry(graph_id1)
                .or_insert_with(|| (Vec::new(), HashSet::new()))
                .0
                .push((node1 - 1, node2 - 1));
            graph_dict
                .get_mut(&graph_id1)
                .unwrap()
                .1
                .extend(vec![node1 - 1, node2 - 1]);
        }
    }

    // Create adjacency matrices and node labels for each graph
    let mut graphs: Vec<Graph> = Vec::new();
    let mut target_labels: Vec<i32> = Vec::new();
    for (graph_id, (edges, nodes)) in graph_dict {
        let mut nodes: Vec<usize> = nodes.into_iter().collect();
        nodes.sort();
        let node_map: FxHashMap<usize, usize> = nodes
            .iter()
            .enumerate()
            .map(|(idx, &node)| (node, idx))
            .collect();
        let n_vertices: usize = nodes.len();

        let mut adjacency_matrix: Vec<Vec<usize>> = vec![vec![0; n_vertices]; n_vertices];
        for (node1, node2) in edges {
            adjacency_matrix[node_map[&node1]][node_map[&node2]] = 1;
            adjacency_matrix[node_map[&node2]][node_map[&node1]] = 1;
        }

        let mut node_index_dict: FxHashMap<usize, i32> = FxHashMap::default();
        for (&node, &index) in node_map.iter() {
            node_index_dict.insert(index, node_labels[node]);
        }

        // Initialize the graph with adjacency matrix and node labels
        let graph: Graph = Graph {
            adjacency_matrix,
            node_index_dict,
            n_vertices,
        };
        graphs.push(graph);
        target_labels.push(graph_labels[graph_id - 1]); // Graph IDs are 1-based, so subtract 1
    }
    // println!("Graph vertices: {:?}", graphs[0].n_vertices);
    // println!("Graph adjacency matrix: {:?}", graphs[0].adjacency_matrix);
    // println!("Graph node labels: {:?}", graphs[0].node_index_dict);
    // println!("Graph target label: {:?}", target_labels[0]);

    Ok((graphs, target_labels))
}

fn train_test_split(
    graphs: Vec<Graph>,
    target_labels: Vec<i32>,
    test_size: f64,
) -> (Vec<Graph>, Vec<Graph>, Vec<i32>, Vec<i32>) {
    let num_samples: usize = graphs.len();
    let num_test: usize = (num_samples as f64 * test_size).round() as usize;
    let num_train: usize = num_samples - num_test;

    let mut indices: Vec<usize> = (0..num_samples).collect();
    indices.shuffle(&mut thread_rng());

    let (test_indices, train_indices) = indices.split_at(num_test);

    let train_graphs = train_indices.iter().map(|&i| graphs[i].clone()).collect();
    let test_graphs = test_indices.iter().map(|&i| graphs[i].clone()).collect();
    let train_labels = train_indices.iter().map(|&i| target_labels[i]).collect();
    let test_labels = test_indices.iter().map(|&i| target_labels[i]).collect();

    (train_graphs, test_graphs, train_labels, test_labels)
}

fn main() {
    // for each folder in ./DATA
    for folder in fs::read_dir("./TEST").unwrap() {
        // let mut kernel: NeighbourhoodHashKernel = NeighbourhoodHashKernel {
        //     labels_hash_dict: FxHashMap::default(),
        //     r: 2,
        //     x: Vec::new(),
        // };

        let mut kernel = WassersteinHashKernel {
            labels_hash_dict: FxHashMap::default(),
            x: Vec::new(),
        };

        let folder_path: String = folder.unwrap().path().display().to_string();
        let folder_name: String = folder_path.split('/').last().unwrap().to_string();

        let folder_path: String = format!("{}/{}", folder_path, folder_name);

        let start_time = std::time::Instant::now();

        let (graphs, target_labels) = read_data(&folder_path).unwrap();

        print!("Graphs: {:?}", graphs.len());

        let folder_read_time = start_time.elapsed().as_secs_f64();

        let kernel_matrix = kernel.fit_transform(graphs.clone());
        let k = kernel.transform(graphs);

        // print kernel matrix
        for i in 0..kernel_matrix.shape()[0] {
            for j in 0..kernel_matrix.shape()[1] {
                print!("{:.3} ", kernel_matrix[[i, j]]);
            }
            println!();
        }

        let dat_fit_time = start_time.elapsed().as_secs_f64() - folder_read_time;

        // let t = kernel.transform(graphs);

        println!(
            "\nFolder: {}\nRead time: {:.3} s\nFit time: {:.3} s",
            folder_name, folder_read_time, dat_fit_time
        );
    }
}
