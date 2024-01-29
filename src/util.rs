use std::cmp;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, ensure, Context, Result};
use crossbeam_channel::Sender;
use log::info;
use nalgebra::{DMatrix, DVector};

use crate::io;
use crate::io::gwas::{GwasResults, IntermediateResults};
use crate::stats::running::RunningSufficientStats;

fn gwas_path_to_phenotype(filename: &str) -> String {
    Path::new(filename)
        .file_name()
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

    info!("Found GWAS result files: {:?}", gwas_result_files);

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

pub struct RuntimeConfig {
    pub num_threads: usize,
    pub chunksize: usize,
    pub compress: bool,
}

fn gwas_reader(
    gwas_result_files: &[String],
    column_names: io::gwas::ColumnSpec,
    start_line: usize,
    end_line: usize,
    num_lines: usize,
    output: Sender<(String, io::gwas::GwasResults)>,
) -> Result<()> {
    let n_files = gwas_result_files.len();
    for (i, filename) in gwas_result_files.iter().enumerate() {
        let phenotype_name = gwas_path_to_phenotype(filename);
        info!(
            "File {} of {}: Reading lines {} to {} of {} in {}. Interpreted phenotype name: {}",
            i, n_files, start_line, end_line, num_lines, filename, phenotype_name
        );

        let gwas_results =
            io::gwas::read_gwas_results(filename, &column_names, start_line, end_line)
                .with_context(|| format!("Error reading GWAS results from file: {}", &filename))
                .unwrap();

        output.send((phenotype_name, gwas_results))?;
    }

    Ok(())
}

pub struct ProcessingStats {
    pub n_variants: usize,
    pub proj: DMatrix<f32>,
    pub fpv: DVector<f32>,
    pub phenotype_id_to_idx: HashMap<String, usize>,
    pub n_covar: usize,
}

impl ProcessingStats {
    pub fn format_update(
        &self,
        phenotype_id: &str,
        gwas_results: &GwasResults,
    ) -> IntermediateResults {
        let phenotype_idx = self.phenotype_id_to_idx[phenotype_id];

        let b = &gwas_results.beta_values;
        let se = &gwas_results.se_values;
        let ss = &gwas_results.sample_sizes;

        let beta_update = b * self.proj.row(phenotype_idx);

        let mut gpv_update = DVector::zeros(self.n_variants);
        for i in 0..self.n_variants {
            gpv_update[i] = self.fpv[phenotype_idx]
                / (se[i].powi(2) * (ss[i] - self.n_covar as i32 - 2) as f32 + b[i].powi(2));
        }

        IntermediateResults {
            beta_update,
            gpv_update,
            sample_sizes: gwas_results.sample_sizes.clone(),
            variant_ids: gwas_results.variant_ids.clone(),
        }
    }
}

fn process_chunk(
    gwas_result_files: Vec<String>,
    column_names: io::gwas::ColumnSpec,
    start_line: usize,
    end_line: usize,
    num_lines: usize,
    output_file: &str,
    runtime_config: &RuntimeConfig,
    running: Arc<Mutex<RunningSufficientStats>>,
) -> Result<()> {
    let processing_stats = Arc::new(running.lock().unwrap().build_processing_stats());

    let (raw_sender, raw_receiver) = crossbeam_channel::unbounded::<(String, GwasResults)>();
    let (fmt_sender, fmt_receiver) = crossbeam_channel::unbounded::<IntermediateResults>();

    let updater = std::thread::spawn({
        let running = running.clone();
        move || {
            let mut running = running.lock().unwrap();
            for intermediate_results in fmt_receiver.iter() {
                running.update(&intermediate_results);
            }
        }
    });

    let mut workers = Vec::new();
    for _ in 0..runtime_config.num_threads {
        let receiver = raw_receiver.clone();
        let sender = fmt_sender.clone();
        let processing_stats = processing_stats.clone();
        workers.push(std::thread::spawn(move || {
            for (phenotype_name, gwas_results) in receiver.iter() {
                let result = processing_stats.format_update(&phenotype_name, &gwas_results);
                sender.send(result).unwrap();
            }
        }));
    }

    let reader = std::thread::spawn({
        let gwas_result_files = gwas_result_files.clone();
        let column_names = column_names.clone();
        let sender = raw_sender.clone();
        move || {
            gwas_reader(
                &gwas_result_files,
                column_names,
                start_line,
                end_line,
                num_lines,
                sender,
            )
        }
    });

    reader.join().unwrap()?;
    drop(raw_sender);
    info!("Finished reading chunk, waiting for workers to finish");

    for worker in workers {
        worker.join().unwrap();
    }
    drop(fmt_sender);

    updater.join().unwrap();
    info!("Finished reading chunk, computing statistics");

    let final_stats = running.lock().unwrap().compute_final_stats();

    info!("Writing results to file: {}", output_file);
    let include_header = start_line == 0;
    io::gwas::write_gwas_results(
        final_stats,
        output_file,
        include_header,
        runtime_config.compress,
    )
    .with_context(|| format!("Error writing GWAS results to file: {}", output_file))?;

    Ok(())
}

pub fn run(
    projection_matrix_path: &str,
    covariance_matrix_path: &str,
    gwas_result_files: &[String],
    output_file: &str,
    num_covar: usize,
    runtime_config: RuntimeConfig,
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
        runtime_config.chunksize,
    )));

    let num_lines = io::gwas::count_lines(&gwas_result_files[0])?;
    let mut start_line = 0;
    let mut end_line = 0;
    while start_line < num_lines {
        end_line = cmp::min(num_lines, end_line + runtime_config.chunksize);

        let new_chunksize = end_line - start_line;
        running.lock().unwrap().clear_chunk(new_chunksize);

        process_chunk(
            gwas_result_files.clone(),
            column_names.clone(),
            start_line,
            end_line,
            num_lines,
            output_file,
            &runtime_config,
            running.clone(),
        )?;

        start_line = end_line;
    }

    Ok(())
}
