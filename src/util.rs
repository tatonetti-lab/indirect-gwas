use std::cmp;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::io;
use crate::stats::running::RunningSufficientStats;

use anyhow::{anyhow, bail, ensure, Context, Result};
use log::info;
use rayon::prelude::*;

fn gwas_path_to_phenotype(filename: &str) -> String {
    Path::new(filename)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string()
}

/// Check that a GWAS result file has been provided for every phenotype in the
/// projection and covariance matrices. Filter out all GWAS result files that
/// are not needed.
fn check_filter_inputs(
    projection_labels: &[String],
    covariance_labels: &[String],
    gwas_result_files: &[String],
) -> Result<Vec<String>> {
    ensure!(
        projection_labels == covariance_labels,
        "Projection and covariance matrices have different labels"
    );

    let mut phenotype_to_gwas_path: HashMap<String, String> = HashMap::new();
    for gwas_path in gwas_result_files {
        let phenotype = gwas_path_to_phenotype(gwas_path);
        if phenotype_to_gwas_path.contains_key(&phenotype) {
            bail!("Multiple GWAS files provided for phenotype {}", phenotype);
        }
        phenotype_to_gwas_path.insert(phenotype, gwas_path.to_string());
    }

    let mut final_gwas_paths = Vec::new();
    for phenotype in projection_labels {
        let path = phenotype_to_gwas_path.remove(phenotype).ok_or(anyhow!(
            "No GWAS result file provided for phenotype {}",
            phenotype
        ))?;
        final_gwas_paths.push(path);
    }

    Ok(final_gwas_paths)
}

pub fn run(
    projection_matrix_path: &str,
    covariance_matrix_path: &str,
    gwas_result_files: &[String],
    output_file: &str,
    num_covar: usize,
    chunksize: usize,
    column_names: io::gwas::ColumnSpec,
) -> Result<()> {
    let projection_matrix =
        io::matrix::read_labeled_matrix(projection_matrix_path).with_context(|| {
            format!(
                "Error reading projection matrix: {}",
                projection_matrix_path
            )
        })?;

    let cov_matrix =
        io::matrix::read_labeled_matrix(covariance_matrix_path).with_context(|| {
            format!(
                "Error reading covariance matrix: {}",
                covariance_matrix_path
            )
        })?;

    info!("Projection shape {:?}", projection_matrix.matrix.shape());
    info!("Covariance shape {:?}", cov_matrix.matrix.shape());
    info!("Covariance labels {:?}", cov_matrix.col_labels);
    info!("Projection labels {:?}", projection_matrix.row_labels);

    let gwas_result_files = check_filter_inputs(
        &projection_matrix.row_labels,
        &cov_matrix.col_labels,
        gwas_result_files,
    )?;

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

        let new_chunksize = end_line - start_line;
        running.lock().unwrap().clear_chunk(new_chunksize);

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
                io::gwas::read_gwas_results(filename, &column_names, start_line, end_line)
                    .with_context(|| format!("Error reading GWAS results from file: {}", &filename))
                    .unwrap();

            running
                .lock()
                .unwrap()
                .update(&phenotype_name, &gwas_results);
        });

        let final_stats = running.lock().unwrap().compute_final_stats();
        let include_header = start_line == 0;
        io::gwas::write_gwas_results(final_stats, output_file, include_header)
            .with_context(|| format!("Error writing GWAS results to file: {}", output_file))?;

        start_line = end_line;
    }

    let final_stats = running.lock().unwrap().compute_final_stats();
    io::gwas::write_gwas_results(final_stats, output_file, false)
        .with_context(|| format!("Error writing GWAS results to file: {}", output_file))?;

    Ok(())
}
