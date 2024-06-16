mod graph;

use crate::graph::Graph;

use core::time;

use ndarray::Array2;
use rustc_hash::FxHashMap;

use std::{
    collections::HashSet,
    fs::{self, File},
    io::{self, BufRead},
};

struct ExampleKernel {
    // The base map of labels to hashes
    labels_hash_dict: FxHashMap<i32, usize>,

    // A vector where each entry is a tuple of (vertex, hashed label, neighbours) for one graph
    x: Vec<(usize, FxHashMap<usize, usize>, FxHashMap<usize, Vec<usize>>)>,
}

trait GraphKernel {
    // Fit a dataset for a transformer
    fn fit(&mut self, graphs: Vec<Graph>);

    // // Fit and transform, on the same dataset
    fn fit_transform(&mut self, graphs: Vec<Graph>) -> Array2<f64>;

    // // Calculate the kernel matrix, between given and fitted dataset
    // fn transform(&self, graphs: &Vec<Graph>) -> Vec<Vec<f64>>;

    // Calculate the kernel matrix given a target_graph and a kernel
    // fn calculate_kernel_matrix(&self) -> Vec<Vec<f64>>;

    // // Calculate a pairwise kernel between two elements
    // fn pairwise_operation(&self, g1: &Graph, g2: &Graph) -> f64;

    // // Calculate the kernel matrix diagonal of the fit/transformed data
    // fn diagonal(&self) -> Vec<f64>;

    fn neighbourhood_hash(
        &self,
        vertex: usize,
        labels: &FxHashMap<usize, i32>,
        neighbours: Vec<usize>,
    ) -> usize;

    fn rot_left(&self, hash: usize, n_bits: usize) -> usize;

    fn compare_labels(
        &self,
        labels1: &FxHashMap<usize, usize>,
        labels2: &FxHashMap<usize, usize>,
    ) -> f64 {
        // sort labels by value
        let mut labels1: Vec<(&usize, &usize)> = labels1.iter().collect();
        labels1.sort_by(|a, b| a.1.cmp(b.1));

        let mut labels2: Vec<(&usize, &usize)> = labels2.iter().collect();
        labels2.sort_by(|a, b| a.1.cmp(b.1));

        let mut count: usize = 1;

        let mut i: usize = 0;
        let mut j: usize = 0;

        while i < labels1.len() && j < labels2.len() {
            if labels1[i].1 == labels2[j].1 {
                count += 1;
                i += 1;
                j += 1;
            } else if labels1[i].1 < labels2[j].1 {
                i += 1;
            } else {
                j += 1;
            }
        }

        count as f64 / ((labels1.len() + labels2.len() - count as usize) as f64)
    }

    // fn pairwise_operation(&self, )

    fn calculate_kernel_matrix(
        &self,
        y: Option<&Vec<(usize, FxHashMap<usize, usize>, FxHashMap<usize, Vec<usize>>)>>,
    ) -> Array2<f64>;

    fn make_symmetric(k: &mut Array2<f64>);
}

impl GraphKernel for ExampleKernel {
    // Fit the dataset
    fn fit(&mut self, graphs: Vec<Graph>) {
        // Gather the unique labels present in the whole dataset (some graphs may only have a small subset of labels)
        let unique_labels: Vec<&i32> = graphs
            .iter()
            .map(|graph| graph.node_index_dict.values())
            .flatten()
            .collect::<HashSet<&i32>>()
            .into_iter()
            .collect();

        // For each unique label, generate a unique random bit hash
        for label in unique_labels {
            let hash: usize = rand::random::<usize>();

            self.labels_hash_dict.insert(*label, hash);
        }

        // For each graph, generate the node embeddings
        for graph in (&graphs).iter() {
            let mut new_labels: FxHashMap<usize, usize> = FxHashMap::default();
            let mut graph_neighbours: FxHashMap<usize, Vec<usize>> = FxHashMap::default();

            for vertex in 0..graph.n_vertices {
                let neighbours: Vec<usize> = graph
                    .adjacency_matrix
                    .get(vertex)
                    .unwrap()
                    .iter()
                    .enumerate()
                    .filter(|(_, &edge)| edge == 1)
                    .map(|(idx, _)| idx)
                    .collect();

                let new_label: usize =
                    self.neighbourhood_hash(vertex, &graph.node_index_dict, neighbours.clone());

                graph_neighbours.insert(vertex, neighbours);
                new_labels.insert(vertex, new_label);
            }

            // Store the graphs and node embeddings
            self.x
                .push((graph.n_vertices, new_labels, graph_neighbours));
        }
    }

    fn fit_transform(&mut self, graphs: Vec<Graph>) -> Array2<f64> {
        self.fit(graphs);

        self.calculate_kernel_matrix(None)
    }

    fn neighbourhood_hash(
        &self,
        vertex: usize,
        labels: &FxHashMap<usize, i32>,
        neighbours: Vec<usize>,
    ) -> usize {
        let mut count_neighbour_labels: FxHashMap<usize, u32> = FxHashMap::default();

        // Count the number of neighbours for each label
        for neighbour in neighbours {
            let neighbour_label: usize = labels.get(&neighbour).unwrap().clone() as usize;

            count_neighbour_labels
                .entry(neighbour_label)
                .and_modify(|e| *e += 1)
                .or_insert(1);
        }

        // Hash the labels as proposed in the paper
        let mut hash: usize = self.rot_left(
            self.labels_hash_dict
                .get(labels.get(&vertex).unwrap())
                .unwrap()
                .clone(),
            1,
        );

        for (label, count) in count_neighbour_labels.iter() {
            hash ^= self.rot_left(
                self.labels_hash_dict
                    .get(&label.clone().try_into().unwrap())
                    .unwrap()
                    .clone(),
                (*count).try_into().unwrap(),
            );
            hash ^= *count as usize;
        }

        hash
    }

    fn calculate_kernel_matrix(
        &self,
        y: Option<&Vec<(usize, FxHashMap<usize, usize>, FxHashMap<usize, Vec<usize>>)>>,
    ) -> Array2<f64> {
        if y == None {
            let mut kernel_matrix: Array2<f64> = Array2::zeros((self.x.len(), self.x.len()));

            let mut cache: Vec<&(usize, FxHashMap<usize, usize>, FxHashMap<usize, Vec<usize>>)> =
                Vec::new();
            for (i, e) in self.x.iter().enumerate() {
                kernel_matrix[(i, i)] = self.compare_labels(&e.1, &e.1);

                for (j, f) in cache.iter().enumerate() {
                    kernel_matrix[(j, i)] = self.compare_labels(&f.1, &e.1);
                }
                cache.push(&e);
            }

            ExampleKernel::make_symmetric(&mut kernel_matrix);

            kernel_matrix
        } else {
            let mut kernel_matrix: Array2<f64> = Array2::zeros((self.x.len(), y.unwrap().len()));

            kernel_matrix
        }
    }

    fn make_symmetric(k: &mut Array2<f64>) {
        let n = k.shape()[0];

        // Create the upper triangular part including the diagonal
        let mut upper_tri = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                upper_tri[[i, j]] = k[[i, j]];
            }
        }

        // Create the upper triangular part excluding the diagonal and transpose it
        let mut upper_tri_no_diag_transposed = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                upper_tri_no_diag_transposed[[j, i]] = k[[i, j]];
            }
        }

        // Add the upper triangular matrix and the transposed upper triangular matrix without diagonal
        *k = &upper_tri + &upper_tri_no_diag_transposed;
    }

    // Rotate the bits of a BitVec to the left
    fn rot_left(&self, hash: usize, n_bits: usize) -> usize {
        // Get the number of bits in usize
        let bits = std::mem::size_of::<usize>() * 8;
        // Ensure the rotation amount is within the range of 0 to bits-1
        let n_bits = n_bits % bits;
        // Perform the left rotation
        (hash << n_bits) | (hash >> (bits - n_bits))
    }
}

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

fn main() {
    let folder_path: &str = "../DATA/MUTAG/MUTAG";

    // for each folder in ./DATA
    for folder in fs::read_dir("../DATA").unwrap() {
        let mut kernel: ExampleKernel = ExampleKernel {
            labels_hash_dict: FxHashMap::default(),
            x: Vec::new(),
        };

        let folder_path: String = folder.unwrap().path().display().to_string();
        let folder_name: String = folder_path.split('/').last().unwrap().to_string();

        let folder_path: String = format!("{}/{}", folder_path, folder_name);

        let start_time = std::time::Instant::now();

        let (graphs, target_label) = read_data(&folder_path).unwrap();

        let folder_read_time = start_time.elapsed().as_secs_f64();

        let kernel_matrix = kernel.fit_transform(graphs);

        // kernel.fit(graphs);

        let dat_fit_time = start_time.elapsed().as_secs_f64() - folder_read_time;

        println!(
            "\nFolder: {}\nRead time: {:.3} s\nFit time: {:.3} s",
            folder_name, folder_read_time, dat_fit_time
        );

        // print kernel matrix size
        // println!("Kernel matrix size: {:?}", kernel_matrix);
        // // print the sparsity of the kernel matrix
        let mut count: usize = 0;
        let mut r = 0;
        for i in 0..kernel_matrix.shape()[0] {
            for j in 0..kernel_matrix.shape()[1] {
                r += 1;
                if kernel_matrix[(i, j)] == 0.0 {
                    count += 1;
                }
            }
        }
        println!("Count: {}", count);
        println!("R: {}", r);
        println!(
            "Sparsity: {:.3}",
            count as f64 / (kernel_matrix.shape()[0] * kernel_matrix.shape()[1]) as f64
        );
    }
}
