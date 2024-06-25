use ndarray::Array2;
use numpy::{PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn transform() -> Py<PyArray2<f64>> {
    let matrix = Array2::<f64>::zeros((178, 178)); 
    Python::with_gil(|py| matrix.to_pyarray(py).to_owned())
}

#[pymodule]
fn rusty_kernel(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transform, m)?)?;
    Ok(())
}
