mod graph;
mod graph_kernel;
mod neighbourhood_hash_kernel;

use crate::graph_kernel::GraphKernel;

use std::collections::HashMap;

use numpy::{PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rustc_hash::FxHashMap;

use crate::{graph::Graph, neighbourhood_hash_kernel::NeighbourhoodHashKernel};

#[pyfunction]
fn transform<'py>(
    py: Python<'py>,
    data: Vec<(PyObject, HashMap<i32, i32>, i32)>,
) -> Py<PyArray2<f64>> {
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

    let mut kernel: NeighbourhoodHashKernel = NeighbourhoodHashKernel {
        labels_hash_dict: FxHashMap::default(),
        x: Vec::new(),
    };

    let matrix = kernel.fit_transform(graphs);
    Python::with_gil(|py| matrix.to_pyarray(py).to_owned())
}

#[pymodule]
fn rusty_kernel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transform, m)?)?;
    Ok(())
}
