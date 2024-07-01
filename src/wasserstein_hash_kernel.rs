use std::collections::HashSet;

use lp_modeler::dsl::*;
use lp_modeler::solvers::{CbcSolver, SolverTrait};
use lp_modeler::{
    constraint,
    dsl::{LpContinuous, LpInteger, LpObjective, LpOperations, LpProblem},
};
use ndarray::Array2;
use numpy::npyffi::objects;
use rustc_hash::FxHashMap;

use crate::{graph::Graph, graph_kernel::GraphKernel};

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

            let total_graphs = self.x.len();
            let mut progress = 0;

            for (i, e) in self.x.iter().enumerate() {
                // println!("Comparing one label");
                kernel_matrix[(i, i)] = self.compare_labels(&e.1, &e.1);

                for (j, f) in cache.iter().enumerate() {
                    kernel_matrix[(j, i)] = self.compare_labels(&f.1, &e.1);
                }
                cache.push(&e);

                progress += 1;
                println!("Progress: {}/{}", progress, total_graphs);
            }

            WassersteinHashKernel::make_symmetric(&mut kernel_matrix);

            kernel_matrix
        } else {
            let mut kernel_matrix: Array2<f64> = Array2::zeros((self.x.len(), y.unwrap().len()));

            kernel_matrix
        }
    }

    fn compare_labels(
        &self,
        labels1: &FxHashMap<usize, usize>,
        labels2: &FxHashMap<usize, usize>,
    ) -> f64 {
        // println!("Comparing labels");

        let n: usize = labels1.len();
        let n_prime: usize = labels2.len();

        let distance_matrix: Array2<f64> = self.compute_distance_matrix(labels1, labels2);

        let mut problem = LpProblem::new("Transport problem", LpObjective::Minimize);
        let mut objective: LpExpression = 0.0.into();

        let mut p: Vec<Vec<LpContinuous>> = Vec::new();

        // Create variables for the transport matrix p
        for i in 0..n {
            let mut row: Vec<LpContinuous> = Vec::new();
            for j in 0..n_prime {
                let var = LpContinuous::new(&format!("p_{}_{}", i, j));
                problem += var.ge(0.0);
                row.push(var);
            }
            p.push(row);
        }

        // Row constraints: sum of each row must be 1/n
        for i in 0..n {
            let mut row: Vec<LpContinuous> = Vec::new();
            let mut row_constraint = LpExpression::from(0.0);
            for j in 0..n_prime {
                row.push(p[i][j].clone());
                row_constraint += p[i][j].clone() * distance_matrix[[i, j]] as f32;
            }
            // let ref row_sum = LpContinuous::new(&format!("row_sum_{}", i));
            problem += row_constraint.equal(1.0 / (n as f32));
        }

        // Column constraints: sum of each column must be 1/n'
        for j in 0..n_prime {
            let mut column: Vec<LpContinuous> = Vec::new();
            let mut column_constraint = LpExpression::from(0.0);
            for i in 0..n {
                column.push(p[i][j].clone());
                column_constraint += p[i][j].clone() * distance_matrix[[i, j]] as f32;
            }
            // let ref column_sum = LpContinuous::new(&format!("column_sum_{}", j));
            problem += column_constraint.equal(1.0 / (n_prime as f32));
        }

        for i in 0..n {
            for j in 0..n_prime {
                let cost = distance_matrix[[i, j]] as f32;
                objective += cost * p[i][j].clone();
            }
        }

        problem.add_objective_expression(&mut objective);

        // println!("Problem:",);
        let solver = CbcSolver::new();

        // println!("Solving problem");

        match solver.run(&problem) {
            Ok(solution) => {
                // println!("Status {:?}", solution.status);
                let mut total_cost = 0.0;
                let mut sum = 0.0;
                for (name, value) in solution.results.iter() {
                    // Parse the variable name to get the indices i and j
                    let parts: Vec<&str> = name.split('_').collect();

                    // println!("name: {}, {}", name, value);
                    sum += *value as f64;
                    if parts.len() == 3 {
                        let i: usize = parts[1].parse().unwrap();
                        let j: usize = parts[2].parse().unwrap();
                        // Compute the transport cost
                        total_cost += distance_matrix[[i, j]] * *value as f64;
                    }
                }
                // println!("Total transport cost = {}", total_cost);
                // println!("Sum = {}", sum);
                return total_cost;
            }
            Err(msg) => {
                println!("{}", msg);

                // panic!("Error solving problem")
                1.0
            }
        }

        // 0.0
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
    fn compute_distance_matrix(
        &self,
        labels1: &FxHashMap<usize, usize>,
        labels2: &FxHashMap<usize, usize>,
    ) -> Array2<f64> {
        // println!("Computing distance matrix between:");

        // for (key, value) in labels1.iter() {
        //     println!("{}: {}", key, value);
        // }

        // for (key, value) in labels2.iter() {
        //     println!("{}: {}", key, value);
        // }

        let n: usize = labels1.len();
        let n_prime: usize = labels2.len();

        let mut distance_matrix: Array2<f64> = Array2::zeros((n, n_prime));

        for i in 0..n {
            for j in 0..n_prime {
                distance_matrix[(i, j)] = self.hamming_distance(
                    labels1.get(&i).unwrap().clone(),
                    labels2.get(&j).unwrap().clone(),
                );

                // println!(
                //     "Hamming distance between {:x} and {:x}: {}",
                //     labels1.get(&i).unwrap(),
                //     labels2.get(&j).unwrap(),
                //     distance_matrix[(i, j)]
                // );
            }
        }

        // print distance matrix
        // for i in 0..n {
        //     for j in 0..n_prime {
        //         print!("{:.3} ", distance_matrix[[i, j]]);
        //     }
        //     println!();
        // }

        distance_matrix
    }

    fn hamming_distance(&self, a: usize, b: usize) -> f64 {
        let bit_count = (a ^ b).count_ones();
        bit_count as f64 / (usize::BITS as f64)
    }
}
