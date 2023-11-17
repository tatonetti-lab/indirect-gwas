use crate::igwas::running::RunningSufficientStats;
use crate::io::gwas::write_gwas_results;

use env_logger;
use log::info;

mod igwas;
mod io;

fn main() {
    let mut builder = env_logger::Builder::from_default_env();
    builder.target(env_logger::Target::Stdout);
    builder.init();

    info!("Starting up IGWAS!");

    // Actual testing
    let projection_matrix = io::matrix::read_labeled_matrix(
        "/Users/zietzm/Documents/projects/indirect_gwas_analysis/data/experiment_2/projection_coef.csv",
    ).unwrap();
    info!(
        "Read projection matrix with shape {:?}",
        projection_matrix.matrix.shape()
    );

    let cov_matrix = io::matrix::read_labeled_matrix(
        "/Users/zietzm/Documents/projects/indirect_gwas_analysis/data/experiment_2/feature_cov.csv",
    )
    .unwrap();
    info!(
        "Read covariance matrix with shape {:?}",
        cov_matrix.matrix.shape()
    );

    let colspec = io::gwas::ColumnSpec {
        variant_id: "variant_id".to_string(),
        beta: "beta".to_string(),
        se: "std_error".to_string(),
        sample_size: "sample_size".to_string(),
    };

    let file_stem =
        "/Users/zietzm/Documents/projects/indirect_gwas_analysis/data/experiment_2/gwas/";

    let mut running = RunningSufficientStats::new(&projection_matrix, &cov_matrix, 12, 100000);

    for i in 0..10 {
        let filename = format!("{}F{}.csv", file_stem, i);

        info!("Reading F{}", i);

        let mut gwas_results = io::gwas::read_gwas_results(&filename, &colspec, 0, 99999).unwrap();
        running.update(&format!("F{}", i), &mut gwas_results);
    }
    let final_stats = running.compute_final_stats();

    let write_result = write_gwas_results(
        final_stats,
        "/Users/zietzm/Documents/projects/igwas_rs/TEST3.csv",
    );

    match write_result {
        Ok(_) => println!("Second write successful!"),
        Err(e) => println!("Error: {}", e),
    }

    info!("Done!");
}
