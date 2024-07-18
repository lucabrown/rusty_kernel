use std::collections::HashMap;

use ndarray::Array2;
use numpy::ToPyArray;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

use crate::{
    graph::Graph, graph_kernel::GraphKernel, neighbourhood_hash_kernel::NeighbourhoodHashKernel,
    wasserstein_hash_kernel::WassersteinHashKernel,
};

#[pyclass]
pub struct PyGraphKernel {
    kernel: WassersteinHashKernel,
}

#[pymethods]
impl PyGraphKernel {
    #[new]
    fn new() -> Self {
        PyGraphKernel {
            kernel: WassersteinHashKernel::new(),
        }
    }

    fn fit_transform(
        &mut self,
        data: Vec<(PyObject, HashMap<i32, i32>, i32)>,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut graphs = Vec::new();

            for (adj_matrix, labels, n) in data {
                let adj_matrix: Vec<Vec<usize>> = adj_matrix.as_ref(py).extract().unwrap();

                // Convert the labels HashMap<i32, i32> into FxHashMap<usize, i32>
                let mut node_index_dict: FxHashMap<usize, i32> = FxHashMap::default();
                for (key, value) in labels {
                    node_index_dict.insert(key as usize, value);
                }

                // Create the graph
                let graph = Graph {
                    adjacency_matrix: adj_matrix,
                    node_index_dict,
                    n_vertices: n as usize,
                };

                graphs.push(graph);
            }

            let matrix: Array2<f64> = self.kernel.fit_transform(graphs);

            let result_converted: PyObject = matrix.to_pyarray(py).to_owned().into();
            Ok(result_converted)
        })
    }

    fn transform(&mut self, data: Vec<(PyObject, HashMap<i32, i32>, i32)>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut graphs = Vec::new();

            for (adj_matrix, labels, n) in data {
                let adj_matrix: Vec<Vec<usize>> = adj_matrix.as_ref(py).extract().unwrap();

                // Convert the labels HashMap<i32, i32> into FxHashMap<usize, i32>
                let mut node_index_dict: FxHashMap<usize, i32> = FxHashMap::default();
                for (key, value) in labels {
                    node_index_dict.insert(key as usize, value);
                }

                // Create the graph
                let graph = Graph {
                    adjacency_matrix: adj_matrix,
                    node_index_dict,
                    n_vertices: n as usize,
                };

                graphs.push(graph);
            }

            let matrix: Array2<f64> = self.kernel.transform(graphs);

            let result_converted: PyObject = matrix.to_pyarray(py).to_owned().into();
            Ok(result_converted)
        })
    }
}
