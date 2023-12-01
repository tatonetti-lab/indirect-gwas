use crate::io::{gwas::GwasResults, gwas::IGwasResults, matrix::LabeledMatrix};
use crate::stats::sumstats::compute_neg_log_pvalue;

use std::collections::HashMap;

use nalgebra::{Const, DMatrix, DVector, Dyn};

pub struct RunningSufficientStats {
    pub beta: DMatrix<f32>,
    pub gpv: DVector<f32>,
    pub sample_sizes: DVector<i32>,

    cov: DMatrix<f32>,  // Partial covariance matrix of the features
    fpv: DVector<f32>,  // Partial variance vector of the features
    proj: DMatrix<f32>, // Matrix of the projection coefficients

    n_covar: usize,
    chunksize: usize,

    n_features: usize,
    n_projections: usize,
    phenotype_id_to_idx: HashMap<String, usize>,

    variant_ids: Option<Vec<String>>,
    projection_ids: Vec<String>,

    n_features_seen: usize,
}

// Add a method on RunningSufficientStats that takes some GWAS summary statistics and updates the
// state
impl RunningSufficientStats {
    pub fn new(
        proj: &LabeledMatrix,
        cov: &LabeledMatrix,
        n_covar: usize,
        chunksize: usize,
    ) -> Self {
        let n_features = proj.matrix.nrows();
        let n_projections = proj.matrix.ncols();

        // Check that cov is n_features x n_features
        assert_eq!(
            cov.matrix.nrows(),
            n_features,
            "Covariance matrix has wrong shape, expected {} x {}, got {} x {}",
            n_features,
            n_features,
            cov.matrix.nrows(),
            cov.matrix.ncols()
        );
        assert_eq!(
            cov.matrix.ncols(),
            n_features,
            "Covariance matrix has wrong shape, expected {} x {}, got {} x {}",
            n_features,
            n_features,
            cov.matrix.nrows(),
            cov.matrix.ncols()
        );

        // Phenotype_id_to_idx is a hashmap basically of an enumeration of the phenotype ids
        let phenotype_id_to_idx = proj
            .row_labels
            .iter()
            .enumerate()
            .map(|(i, x)| (x.clone(), i))
            .collect();

        RunningSufficientStats {
            beta: DMatrix::zeros(chunksize, n_projections),
            gpv: DVector::zeros(chunksize),
            sample_sizes: DVector::zeros(chunksize),
            cov: cov.matrix.clone(),
            fpv: cov.matrix.diagonal(),
            proj: proj.matrix.clone(),
            n_covar,
            n_features,
            n_projections,
            chunksize,
            phenotype_id_to_idx,
            variant_ids: None,
            projection_ids: proj.col_labels.clone(),
            n_features_seen: 0,
        }
    }
    pub fn update(&mut self, phenotype_id: &str, gwas_results: &GwasResults) {
        if self.n_features_seen == 0 {
            self.sample_sizes = gwas_results.sample_sizes.clone();
            self.variant_ids = Some(gwas_results.variant_ids.clone());
        } else {
            self.sample_sizes = self.sample_sizes.inf(&gwas_results.sample_sizes);
            // Check that the variant ids match. Add a runtime error message if not
            assert_eq!(
                self.variant_ids.clone().unwrap(),
                gwas_results.variant_ids,
                "Mismatched variant ids"
            );
        }

        let phenotype_idx = self.phenotype_id_to_idx[phenotype_id];
        let coef = &self.proj.row(phenotype_idx);

        let b = &gwas_results.beta_values;
        let se = &gwas_results.se_values;
        let ss = &gwas_results.sample_sizes;

        self.beta += b * coef;

        self.gpv += DMatrix::from_fn(self.gpv.nrows(), 1, |i, _| {
            self.fpv[phenotype_idx]
                / (se[i].powi(2) * (ss[i] - self.n_covar as i32 - 2) as f32 + b[i].powi(2))
        });

        self.n_features_seen += 1;
    }

    pub fn compute_final_stats(&mut self) -> IGwasResults {
        if self.n_features_seen != self.n_features {
            panic!(
                "Too few features seen. Expected {}, got {}",
                self.n_features, self.n_features_seen
            );
        }

        self.gpv /= self.n_features_seen as f32;
        let dof = self.sample_sizes.map(|x| x - 2 - self.n_covar as i32);
        let ppv = (self.proj.transpose() * &self.cov * &self.proj).diagonal();
        let se = DMatrix::from_fn(self.gpv.nrows(), ppv.nrows(), |i, j| {
            ((ppv[j] / self.gpv[i] - self.beta[(i, j)].powi(2)) / dof[i] as f32).sqrt()
        });
        let t_stat = self.beta.component_div(&se);
        let p_values = DMatrix::from_fn(t_stat.nrows(), t_stat.ncols(), |i, j| {
            compute_neg_log_pvalue(t_stat[(i, j)], dof[i])
        });

        let n_elements = self.beta.nrows() * self.beta.ncols();

        let variant_ids: Vec<String> = std::iter::repeat(self.variant_ids.clone().unwrap())
            .take(self.n_projections)
            .flatten()
            .collect();

        let sample_sizes: DVector<i32> = DVector::from_vec(
            self.sample_sizes
                .as_slice()
                .iter()
                .cycle()
                .take(n_elements)
                .cloned()
                .collect(),
        );

        let projection_ids: Vec<String> = self
            .projection_ids
            .iter()
            .flat_map(|x| std::iter::repeat(x.clone()).take(self.chunksize))
            .collect();

        IGwasResults {
            projection_ids,
            variant_ids,
            beta_values: self
                .beta
                .clone()
                .reshape_generic(Dyn(n_elements), Const::<1>),
            se_values: se.reshape_generic(Dyn(n_elements), Const::<1>),
            t_stat_values: t_stat.reshape_generic(Dyn(n_elements), Const::<1>),
            p_values: p_values.reshape_generic(Dyn(n_elements), Const::<1>),
            sample_sizes,
        }
    }
}
