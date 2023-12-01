use std::path::Path;

use igwas::{stats::sumstats::compute_neg_log_pvalue, InputArguments};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub struct TestData {
    pub variant_ids: Vec<String>,
    pub phenotype_ids: Vec<String>,
    pub projection_ids: Vec<String>,
    pub genotypes: DMatrix<f32>,
    pub phenotypes: DMatrix<f32>,
    pub covariates: DMatrix<f32>,
    pub projection_matrix: DMatrix<f32>,
    pub covariance_matrix: DMatrix<f32>,
}

fn residualize(x: &DMatrix<f32>, covariates: &DMatrix<f32>) -> DMatrix<f32> {
    let covariates_intercept = covariates.clone().insert_column(0, 1.0);
    let beta = covariates_intercept
        .clone()
        .svd(true, true)
        .solve(&x, 1e-7)
        .expect("SVD failed");
    x - covariates_intercept * beta
}

fn compute_partial_covariance_matrix(
    phenotypes: &DMatrix<f32>,
    covariates: &DMatrix<f32>,
) -> DMatrix<f32> {
    let phenotype_residuals = residualize(phenotypes, covariates);
    (&phenotype_residuals.transpose() * &phenotype_residuals)
        * (1.0 / (phenotypes.nrows() as f32 - 1.0))
}

// Simulate
fn build_test(
    n_samples: usize,
    n_variants: usize,
    n_covariates: usize,
    n_phenotypes: usize,
    n_projections: usize,
) -> TestData {
    let genotypes = DMatrix::<u8>::new_random(n_samples, n_variants).map(|x| x as f32);
    let phenotypes = DMatrix::<f32>::new_random(n_samples, n_phenotypes);
    let covariates = DMatrix::<f32>::new_random(n_samples, n_covariates);
    let projection_matrix = DMatrix::<f32>::new_random(n_phenotypes, n_projections);
    let covariance_matrix = compute_partial_covariance_matrix(&phenotypes, &covariates);

    TestData {
        variant_ids: (0..n_variants).map(|x| format!("variant_{}", x)).collect(),
        phenotype_ids: (0..n_phenotypes)
            .map(|x| format!("phenotype_{}", x))
            .collect(),
        projection_ids: (0..n_projections)
            .map(|x| format!("projection_{}", x))
            .collect(),
        genotypes,
        phenotypes,
        covariates,
        projection_matrix,
        covariance_matrix,
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct GwasResults {
    pub phenotype_id: String,
    pub variant_id: String,
    pub beta: f32,
    pub std_error: f32,
    pub t_stat: f32,
    pub p_value: f32,
    pub sample_size: usize,
}

fn single_gwas_regression(
    phenotype_id: String,
    variant_id: String,
    genotype_residuals: DVector<f32>,
    phenotypes_residuals: DVector<f32>,
    num_covariates: usize,
) -> GwasResults {
    let gt2 = genotype_residuals.map(|x| x.powi(2)).sum();
    let beta = genotype_residuals.dot(&phenotypes_residuals) / gt2;

    let resids = phenotypes_residuals - genotype_residuals * beta;
    let n = resids.nrows() as f32;

    let std_err =
        (resids.map(|x| x.powi(2)).sum() / (gt2 * (n - num_covariates as f32 - 2.0))).sqrt();
    let t_stat = beta / std_err;
    let p_value = compute_neg_log_pvalue(t_stat, n as i32 - num_covariates as i32 - 2);

    GwasResults {
        phenotype_id,
        variant_id,
        beta,
        std_error: std_err,
        t_stat,
        p_value,
        sample_size: n as usize,
    }
}

fn gwas(
    phenotypes: &DMatrix<f32>,
    genotypes: &DMatrix<f32>,
    covariates: &DMatrix<f32>,
    phenotype_ids: &Vec<String>,
    variant_ids: &Vec<String>,
) -> Vec<Vec<GwasResults>> {
    let phenotypes_residuals = residualize(phenotypes, covariates);
    let genotype_residuals = residualize(genotypes, covariates);

    phenotypes_residuals
        .par_column_iter()
        .enumerate()
        .map(|(phenotype_idx, phenotype_residuals)| {
            genotype_residuals
                .par_column_iter()
                .enumerate()
                .map(|(var_idx, genotype_residuals)| {
                    single_gwas_regression(
                        phenotype_ids[phenotype_idx].clone(),
                        variant_ids[var_idx].clone(),
                        genotype_residuals.into(),
                        phenotype_residuals.into(),
                        covariates.ncols(),
                    )
                })
                .collect::<Vec<GwasResults>>()
        })
        .collect()
}

fn write_gwas_results(results: &Vec<GwasResults>, output_path: &str) {
    let mut writer = csv::Writer::from_path(output_path).unwrap();
    for result in results {
        writer.serialize(result).unwrap();
    }
    writer.flush().unwrap();
}

fn write_test_data(test_data: &TestData, dir: &Path) -> Result<(), std::io::Error> {
    let mut writer = csv::Writer::from_path(dir.join("genotypes.csv"))?;
    for row in test_data.genotypes.row_iter() {
        writer.write_record(row.iter().map(|x| x.to_string()))?;
    }
    writer.flush()?;

    let mut writer = csv::Writer::from_path(dir.join("phenotypes.csv"))?;
    for row in test_data.phenotypes.row_iter() {
        writer.write_record(row.iter().map(|x| x.to_string()))?;
    }
    writer.flush()?;

    let mut writer = csv::Writer::from_path(dir.join("covariates.csv"))?;
    for row in test_data.covariates.row_iter() {
        writer.write_record(row.iter().map(|x| x.to_string()))?;
    }
    writer.flush()?;

    let mut writer = csv::Writer::from_path(dir.join("projection_matrix.csv"))?;
    let mut header_row = test_data.projection_ids.clone();
    header_row.insert(0, "phenotype_id".to_string());
    writer.write_record(header_row)?;
    for (idx, row) in test_data.projection_matrix.row_iter().enumerate() {
        let mut row_vec = row.iter().map(|x| x.to_string()).collect::<Vec<String>>();
        row_vec.insert(0, test_data.phenotype_ids[idx].clone());
        writer.write_record(row_vec)?;
    }
    writer.flush()?;

    let mut writer = csv::Writer::from_path(dir.join("covariance_matrix.csv"))?;
    let mut header_row = test_data.phenotype_ids.clone();
    header_row.insert(0, "phenotype_id".to_string());
    writer.write_record(header_row)?;
    for (idx, row) in test_data.covariance_matrix.row_iter().enumerate() {
        let mut row_vec = row.iter().map(|x| x.to_string()).collect::<Vec<String>>();
        row_vec.insert(0, test_data.phenotype_ids[idx].clone());
        writer.write_record(row_vec)?;
    }
    writer.flush()?;
    Ok(())
}

pub fn setup_test(
    dir: &Path,
    n_samples: usize,
    n_variants: usize,
    n_covariates: usize,
    n_phenotypes: usize,
    n_projections: usize,
) -> InputArguments {
    std::fs::create_dir_all(dir.join("gwas")).unwrap();

    let test_data = build_test(
        n_samples,
        n_variants,
        n_covariates,
        n_phenotypes,
        n_projections,
    );
    write_test_data(&test_data, dir).unwrap();

    let do_gwas =
        |phenotypes: &DMatrix<f32>, phenotype_ids: &Vec<String>| -> Vec<Vec<GwasResults>> {
            gwas(
                phenotypes,
                &test_data.genotypes,
                &test_data.covariates,
                phenotype_ids,
                &test_data.variant_ids,
            )
        };

    // Do and write feature GWAS
    let feature_gwas_results = do_gwas(&test_data.phenotypes, &test_data.phenotype_ids);

    let feature_gwas_paths: Vec<String> = feature_gwas_results
        .par_iter()
        .enumerate()
        .map(|(idx, results)| {
            let path = dir.join("gwas").join(test_data.phenotype_ids[idx].clone())
                .with_extension("csv").to_str().unwrap().to_string();
            write_gwas_results(results, &path);
            path
        })
        .collect();

    // Do and write projection GWAS
    let projection_phenotypes = &test_data.phenotypes * &test_data.projection_matrix;

    let projection_gwas_results = do_gwas(&projection_phenotypes, &test_data.projection_ids)
        .into_iter()
        .flatten()
        .collect::<Vec<GwasResults>>();

    write_gwas_results(
        &projection_gwas_results,
        &dir.join("direct_results.csv").to_str().unwrap().to_string(),
    );

    InputArguments {
        projection_matrix: dir.join("projection_matrix.csv").to_str().unwrap().to_string(),
        covariance_matrix: dir.join("covariance_matrix.csv").to_str().unwrap().to_string(),
        gwas_results: feature_gwas_paths,
        output_file: dir.join("igwas_results.csv").to_str().unwrap().to_string(),
        num_covar: n_covariates,
        chunksize: n_variants,
        variant_id: "variant_id".to_string(),
        beta: "beta".to_string(),
        std_error: "std_error".to_string(),
        sample_size: "sample_size".to_string(),
        quiet: true,
    }
}

fn read_igwas_results(path: &str) -> Vec<GwasResults> {
    let mut reader = csv::Reader::from_path(path).unwrap();
    let mut results = Vec::new();
    for result in reader.deserialize() {
        let result = result.unwrap();
        results.push(result);
    }
    results
}

pub fn check_results(direct_results_path: &str, indirect_results_path: &str) {
    let direct_results = read_igwas_results(direct_results_path);
    let igwas_results = read_igwas_results(indirect_results_path);

    let tol = 1e-4;

    for (direct, igwas) in direct_results.iter().zip(igwas_results.iter()) {
        assert_eq!(direct.variant_id, igwas.variant_id);
        assert_eq!(direct.phenotype_id, igwas.phenotype_id);

        assert!(
            (direct.beta - igwas.beta).abs() < tol,
            "Beta - Direct: {}, Indirect: {}",
            direct.beta,
            igwas.beta
        );
        assert!(
            (direct.std_error - igwas.std_error).abs() < tol,
            "SE - Direct: {}, Indirect: {}",
            direct.std_error,
            igwas.std_error
        );
        assert!(
            (direct.t_stat - igwas.t_stat).abs() < tol,
            "T - Direct: {}, Indirect: {}",
            direct.t_stat,
            igwas.t_stat
        );
        assert!(
            (direct.p_value - igwas.p_value).abs() < tol,
            "P - Direct: {}, Indirect: {}",
            direct.p_value,
            igwas.p_value
        );
        assert_eq!(
            direct.sample_size, igwas.sample_size,
            "Sample size - Direct: {}, Indirect: {}",
            direct.sample_size, igwas.sample_size
        );
    }
}
