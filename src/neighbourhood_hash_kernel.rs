use std::collections::HashSet;

use ndarray::Array2;
use rustc_hash::FxHashMap;

use crate::{graph::Graph, graph_kernel::GraphKernel};

pub struct NeighbourhoodHashKernel {
    // The base map of labels to hashes
    pub labels_hash_dict: FxHashMap<i32, usize>,

    // The number of hash cycles
    pub r: usize,

    // A vector where each entry is a tuple of (vertex, hashed label, neighbours) for one graph
    pub x: Vec<(usize, FxHashMap<usize, usize>, FxHashMap<usize, Vec<usize>>)>,
}

impl GraphKernel for NeighbourhoodHashKernel {
    fn new() -> Self {
        NeighbourhoodHashKernel {
            labels_hash_dict: FxHashMap::default(),
            r: 1,
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
        for label in unique_labels {
            let hash: usize = rand::random::<usize>();

            self.labels_hash_dict.insert(*label, hash);
        }

        // For each graph, generate the node embeddings
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
            self.x
                .push((graph.n_vertices, new_labels, graph_neighbours));
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

    fn fit_transform(&mut self, graphs: Vec<Graph>) -> Array2<f64> {
        self.fit(graphs);

        self.calculate_kernel_matrix(None)
    }

    // Calculate the kernel matrix, between given and fitted dataset
    fn transform(&self, graphs: Vec<Graph>) -> Array2<f64> {
        if self.x.is_empty() {
            panic!("The kernel has not been fitted yet");
        }

        let mut y: Vec<(usize, FxHashMap<usize, usize>, FxHashMap<usize, Vec<usize>>)> = Vec::new();

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
            for (i, e) in self.x.iter().enumerate() {
                kernel_matrix[(i, i)] = self.compare_labels(&e.1, &e.1);

                for (j, f) in cache.iter().enumerate() {
                    kernel_matrix[(j, i)] = self.compare_labels(&f.1, &e.1);
                }
                cache.push(&e);
            }

            NeighbourhoodHashKernel::make_symmetric(&mut kernel_matrix);

            kernel_matrix
        } else {
            let mut kernel_matrix: Array2<f64> = Array2::zeros((self.x.len(), y.unwrap().len()));

            for (i, e) in self.x.iter().enumerate() {
                for (j, f) in y.unwrap().iter().enumerate() {
                    kernel_matrix[(i, j)] = self.compare_labels(&e.1, &f.1);
                }
            }

            // print!("{:?}", kernel_matrix);

            kernel_matrix
        }
    }

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
