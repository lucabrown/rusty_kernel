mod graph;
mod graph_kernel;
mod neighbourhood_hash_kernel;
mod py_graph_kernel;

use py_graph_kernel::PyGraphKernel;
use pyo3::prelude::*;

#[pymodule]
fn rusty_kernel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGraphKernel>()?;
    Ok(())
}
