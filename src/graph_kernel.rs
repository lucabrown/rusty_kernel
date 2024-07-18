use ndarray::Array2;
use rustc_hash::FxHashMap;

use crate::graph::Graph;

pub(crate) trait GraphKernel {
    #[allow(dead_code)]
    fn new() -> Self;

    // Fit a dataset for a transformer
    fn fit(&mut self, graphs: Vec<Graph>);

    // Fit and transform, on the same dataset
    fn fit_transform(&mut self, graphs: Vec<Graph>) -> Array2<f64>;

    //  Calculate the kernel matrix, between given and fitted dataset
    #[allow(dead_code)]
    fn transform(&mut self, graphs: Vec<Graph>) -> Array2<f64>;

    // Calculate the count sensitive neighbourhood hash of the fitted dataset
    fn neighbourhood_hash(
        &self,
        vertex: usize,
        labels: &FxHashMap<usize, i32>,
        neighbours: Vec<usize>,
    ) -> usize;

    // Returns the kernel value between two labels
    fn compare_labels(
        &self,
        labels1: &FxHashMap<usize, usize>,
        labels2: &FxHashMap<usize, usize>,
    ) -> f64;

    // Calculate the kernel matrix given a target_graph and a kernel
    fn calculate_kernel_matrix(
        &self,
        y: Option<&Vec<(usize, FxHashMap<usize, usize>, FxHashMap<usize, Vec<usize>>)>>,
    ) -> Array2<f64>;

    // Shifts a bit string left by n_bits
    fn rot_left(&self, hash: usize, n_bits: usize) -> usize;

    // Copies the upper triangle of a matrix to the lower triangle
    fn make_symmetric(k: &mut Array2<f64>);

    // Returns the neighbours of a vertex in a graph
    fn get_neighbours(&self, graph: &Graph, vertex: usize) -> Vec<usize>;
}
