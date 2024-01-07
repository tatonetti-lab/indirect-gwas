use std::cmp;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::io;
use crate::stats::running::RunningSufficientStats;

use anyhow::{Context, Result};
use log::info;
use rayon::prelude::*;

pub fn run(
    projection_matrix_path: &str,
    covariance_matrix_path: &str,
    gwas_result_files: &Vec<String>,
    output_file: &str,
    num_covar: usize,
    chunksize: usize,
    variant_id: String,
    beta: String,
    std_error: String,
    sample_size: String,
) -> Result<()> {
    let projection_matrix =
        io::matrix::read_labeled_matrix(projection_matrix_path).with_context(|| {
            format!(
                "Error reading projection matrix: {}",
                projection_matrix_path
            )
        })?;

    info!(
        "Projection matrix has shape {:?}",
        projection_matrix.matrix.shape()
    );

    let cov_matrix =
        io::matrix::read_labeled_matrix(covariance_matrix_path).with_context(|| {
            format!(
                "Error reading covariance matrix: {}",
                covariance_matrix_path
            )
        })?;

    let colspec = io::gwas::ColumnSpec {
        variant_id,
        beta,
        se: std_error,
        sample_size,
    };

    let running = Arc::new(Mutex::new(RunningSufficientStats::new(
        &projection_matrix,
        &cov_matrix,
        num_covar,
        chunksize,
    )));

    let num_lines = io::gwas::count_lines(&gwas_result_files[0])?;
    let mut start_line = 0;
    let mut end_line = 0;
    while start_line < num_lines {
        end_line = cmp::min(num_lines, end_line + chunksize);
        gwas_result_files.par_iter().for_each(|filename: &String| {
            let phenotype_name = Path::new(filename)
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();

            info!(
                "Reading lines {} to {} of {} in {}. Interpreted phenotype name: {}",
                start_line, end_line, num_lines, filename, phenotype_name
            );

            let gwas_results =
                io::gwas::read_gwas_results(filename, &colspec, start_line, end_line)
                    .with_context(|| format!("Error reading GWAS results from file: {}", &filename))
                    .unwrap();

            running
                .lock()
                .unwrap()
                .update(&phenotype_name, &gwas_results);
        });

        start_line = end_line;
    }

    let final_stats = running.lock().unwrap().compute_final_stats();

    io::gwas::write_gwas_results(final_stats, output_file)
        .with_context(|| format!("Error writing GWAS results to file: {}", output_file))?;

    Ok(())
}
