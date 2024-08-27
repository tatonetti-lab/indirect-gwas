use pyo3::{exceptions::PyRuntimeError, prelude::*};

#[pymodule]
fn _lowlevel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(igwas_impl, m)?)?;
    Ok(())
}

#[pyfunction]
fn igwas_impl(
    projection_matrix: String,
    covariance_matrix: String,
    gwas_results: Vec<String>,
    output_file: String,
    num_covar: usize,
    chunksize: usize,
    variant_id: String,
    beta: String,
    std_error: String,
    sample_size: String,
    num_threads: usize,
    capacity: usize,
    compress: bool,
    quiet: bool,
) -> PyResult<()> {
    let args = crate::cli::InputArguments {
        projection_matrix,
        covariance_matrix,
        gwas_results,
        output_file,
        num_covar,
        chunksize,
        variant_id,
        beta,
        std_error,
        sample_size,
        num_threads,
        capacity,
        compress,
        quiet,
        write_phenotype_id: false,
    };
    match crate::cli::run_cli(args) {
        Ok(_) => Ok(()),
        Err(e) => Err(PyRuntimeError::new_err(format!("{e}"))),
    }
}
