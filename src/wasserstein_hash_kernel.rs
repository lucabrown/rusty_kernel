use core::panic;
use std::collections::HashSet;

use ndarray::{Array1, Array2};
use ot::prelude::*;
use rust_optimal_transport as ot;
use rustc_hash::FxHashMap;

use crate::{graph::Graph, graph_kernel::GraphKernel};

/// The WassersteinHashKernel struct represents a kernel that calculates the Wasserstein distance between two graphs
pub struct WassersteinHashKernel {
    // The base map of labels to hashes
    pub labels_hash_dict: FxHashMap<i32, usize>,

    // A vector where each entry is a tuple of (vertex, hashed label, neighbours) for one graph
    pub x: Vec<(usize, FxHashMap<usize, usize>, FxHashMap<usize, Vec<usize>>)>,
}

impl GraphKernel for WassersteinHashKernel {
    fn new() -> Self {
        WassersteinHashKernel {
            labels_hash_dict: FxHashMap::default(),
            x: Vec::new(),
        }
    }

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
        for label in unique_labels.clone() {
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

    // Calculate the kernel matrix, between given and fitted dataset
    fn transform(&mut self, graphs: Vec<Graph>) -> Array2<f64> {
        if self.x.is_empty() {
            panic!("The kernel has not been fitted yet");
        }

        let mut y: Vec<(usize, FxHashMap<usize, usize>, FxHashMap<usize, Vec<usize>>)> = Vec::new();

        let unique_labels: Vec<&i32> = graphs
            .iter()
            .map(|graph| graph.node_index_dict.values())
            .flatten()
            .collect::<HashSet<&i32>>()
            .into_iter()
            .collect();

        // For each unique label, generate a unique random bit hash
        for label in unique_labels.clone() {
            if self.labels_hash_dict.get(label) == None {
                let hash: usize = rand::random::<usize>();

                self.labels_hash_dict.insert(*label, hash);
            }
        }

        for graph in (&graphs).iter() {
            let mut new_labels: FxHashMap<usize, usize> = FxHashMap::default();
            let mut graph_neighbours: FxHashMap<usize, Vec<usize>> = FxHashMap::default();

            for vertex in 0..graph.n_vertices {
                let neighbours: Vec<usize> = self.get_neighbours(graph, vertex);

                let new_label: usize =
                    self.neighbourhood_hash(vertex, &graph.node_index_dict, neighbours.clone());

                graph_neighbours.insert(vertex, neighbours);
                new_labels.insert(vertex, new_label);
            }

            // Store the graphs and node embeddings
            y.push((graph.n_vertices, new_labels, graph_neighbours));
        }

        self.calculate_kernel_matrix(Some(&y))
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

            let mut max_value = std::f64::MIN;
            let mut min_value = std::f64::MAX;

            for (i, e) in self.x.iter().enumerate() {
                // println!("Comparing one label");
                kernel_matrix[(i, i)] = self.compare_labels(&e.1, &e.1);

                if kernel_matrix[(i, i)] > max_value {
                    max_value = kernel_matrix[(i, i)];
                }

                if kernel_matrix[(i, i)] < min_value {
                    min_value = kernel_matrix[(i, i)];
                }

                for (j, f) in cache.iter().enumerate() {
                    kernel_matrix[(j, i)] = self.compare_labels(&f.1, &e.1);

                    if kernel_matrix[(j, i)] > max_value {
                        max_value = kernel_matrix[(j, i)];
                    }

                    if kernel_matrix[(j, i)] < min_value {
                        min_value = kernel_matrix[(j, i)];
                    }
                }
                cache.push(&e);
            }

            WassersteinHashKernel::make_symmetric(&mut kernel_matrix);

            // Normalize the kernel matrix
            for i in 0..self.x.len() {
                for j in 0..self.x.len() {
                    kernel_matrix[(i, j)] =
                        (kernel_matrix[(i, j)] - min_value) / (max_value - min_value);
                }
            }

            kernel_matrix
        } else {
            let mut kernel_matrix: Array2<f64> = Array2::zeros((self.x.len(), y.unwrap().len()));

            let mut max_value = std::f64::MIN;
            let mut min_value = std::f64::MAX;

            for (i, e) in self.x.iter().enumerate() {
                for (j, f) in y.unwrap().iter().enumerate() {
                    kernel_matrix[(i, j)] = self.compare_labels(&e.1, &f.1);

                    if kernel_matrix[(i, j)] > max_value {
                        max_value = kernel_matrix[(i, j)];
                    }
                    if kernel_matrix[(i, j)] < min_value {
                        min_value = kernel_matrix[(i, j)];
                    }
                }
            }

            for i in 0..self.x.len() {
                for j in 0..y.unwrap().len() {
                    kernel_matrix[(i, j)] =
                        (kernel_matrix[(i, j)] - min_value) / (max_value - min_value);
                }
            }

            kernel_matrix
        }
    }

    fn get_neighbours(&self, graph: &Graph, vertex: usize) -> Vec<usize> {
        graph
            .adjacency_matrix
            .get(vertex)
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(_, &edge)| edge == 1)
            .map(|(idx, _)| idx)
            .collect()
    }

    // Returns the Frobenius dot product between distance matrix and transport matrix
    fn compare_labels(
        &self,
        labels1: &FxHashMap<usize, usize>,
        labels2: &FxHashMap<usize, usize>,
    ) -> f64 {
        let n: usize = labels1.len();
        let n_prime: usize = labels2.len();

        let distance_matrix: Array2<f64> = self.compute_distance_matrix(labels1, labels2);

        let epsilon = 1e-6;
        let adjusted_distance_matrix = distance_matrix.mapv(|x| if x == 0.0 { epsilon } else { x });

        // let n = distance_matrix.shape()[0];
        let mut source_weights = Array1::<f64>::from_elem(n, 1.0 / (n as f64));
        let mut target_weights = Array1::<f64>::from_elem(n_prime, 1.0 / (n_prime as f64));

        // Normalize the distance matrix for numerical stability
        let mut max_cost = 0.0;

        for i in 0..n {
            for j in 0..n_prime {
                if adjusted_distance_matrix[[i, j]] > max_cost {
                    max_cost = adjusted_distance_matrix[[i, j]];
                }
            }
        }

        let mut normalized_cost = &adjusted_distance_matrix / max_cost;

        // print the distance matrix and wource and target weight
        // println!("Distance matrix: {:?}", adjusted_distance_matrix);
        // println!("Source weights: {:?}", source_weights);
        // println!("Target weights: {:?}", target_weights);

        assert!(
            (source_weights.sum() - 1.0).abs() < 1e-6,
            "Source weights do not sum to 1."
        );
        assert!(
            (target_weights.sum() - 1.0).abs() < 1e-6,
            "Target weights do not sum to 1."
        );

        let transport_matrix = EarthMovers::new(
            &mut source_weights,
            &mut target_weights,
            &mut normalized_cost,
        )
        .solve()
        .expect("Failed to solve the optimal transport problem");

        let wasserstein_distance: f64 = (&adjusted_distance_matrix * &transport_matrix).sum();

        let laplacian_kernel = self.compute_laplacian_kernel(wasserstein_distance, 20.0);

        laplacian_kernel
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

impl WassersteinHashKernel {
    fn compute_laplacian_kernel(&self, wasserstein_distance: f64, lambda: f64) -> f64 {
        (-lambda * wasserstein_distance).exp()
    }

    fn compute_distance_matrix(
        &self,
        labels1: &FxHashMap<usize, usize>,
        labels2: &FxHashMap<usize, usize>,
    ) -> Array2<f64> {
        let n: usize = labels1.len();
        let n_prime: usize = labels2.len();

        let mut distance_matrix: Array2<f64> = Array2::zeros((n, n_prime));

        for i in 0..n {
            for j in 0..n_prime {
                distance_matrix[(i, j)] = self.hamming_distance(
                    labels1.get(&i).unwrap().clone(),
                    labels2.get(&j).unwrap().clone(),
                );
            }
        }

        distance_matrix
    }

    fn hamming_distance(&self, a: usize, b: usize) -> f64 {
        let bit_count = (a ^ b).count_ones();
        bit_count as f64 / (usize::BITS as f64)
    }
}
