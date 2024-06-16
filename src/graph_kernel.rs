use ndarray::Array2;
use rustc_hash::FxHashMap;

use crate::graph::Graph;

pub(crate) trait GraphKernel {
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
