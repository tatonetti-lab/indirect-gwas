use assert_cmd::prelude::*;
use std::process::Command;
use tempfile::tempdir;

mod utils;

#[test]
fn cli_help() {
    let mut cmd = Command::cargo_bin("igwas").unwrap();
    cmd.arg("--help");
    cmd.assert().success();

    let mut cmd = Command::cargo_bin("igwas").unwrap();
    cmd.arg("--version");
    cmd.assert().success();
}

#[test]
fn read_projection_matrix() {
    let dir = tempdir().unwrap();
    let path = dir.path();
    let args = utils::setup_test(path, 101, 1000, 10, 100, 10);

    let result = igwas::io::matrix::read_labeled_matrix(&args.projection_matrix);

    if let Err(e) = result {
        panic!("Error reading projection matrix: {}", e);
    }

    assert!(result.is_ok());
}

#[test]
fn read_gwas_results() {
    let dir = tempdir().unwrap();
    let path = dir.path();
    let args = utils::setup_test(path, 100, 1000, 10, 100, 10);

    let colnames = igwas::io::gwas::ColumnSpec {
        variant_id: args.variant_id,
        beta: args.beta,
        se: args.std_error,
        sample_size: args.sample_size,
    };

    let result = igwas::io::gwas::read_gwas_results(&args.gwas_results[0], &colnames, 0, 100);

    if let Err(e) = result {
        panic!("Error reading GWAS results: {}", e);
    }

    assert!(result.is_ok());
}

#[test]
fn run_fn() {
    let dir = tempdir().unwrap();
    let path = dir.path();
    let args = utils::setup_test(path, 100, 1000, 10, 100, 10);

    let result = igwas::util::run(
        &args.projection_matrix,
        &args.covariance_matrix,
        &args.gwas_results,
        &args.output_file,
        args.num_covar,
        args.chunksize,
        igwas::io::gwas::ColumnSpec {
            variant_id: args.variant_id,
            beta: args.beta,
            se: args.std_error,
            sample_size: args.sample_size,
        },
    );

    if let Err(e) = result {
        panic!("Error running IGWAS: {}", e);
    }

    assert!(result.is_ok());

    utils::check_results(
        path.join("igwas_results.csv").to_str().unwrap(),
        path.join("direct_results.csv").to_str().unwrap(),
    );
}

#[test]
fn test_run_cli() {
    let dir = tempdir().unwrap();
    let path = dir.path();
    let args = utils::setup_test(path, 100, 1000, 10, 100, 10);

    let result = igwas::run_cli(args);

    if let Err(e) = result {
        panic!("Error running IGWAS: {}", e);
    }

    assert!(result.is_ok());

    utils::check_results(
        path.join("igwas_results.csv").to_str().unwrap(),
        path.join("direct_results.csv").to_str().unwrap(),
    );
}

#[test]
fn cli_full() {
    let dir = tempdir().unwrap();
    let path = dir.path();
    let args = utils::setup_test(path, 100, 1000, 10, 100, 10);

    let mut cmd = Command::cargo_bin("igwas").unwrap();
    cmd.arg("-p")
        .arg(&args.projection_matrix)
        .arg("-c")
        .arg(&args.covariance_matrix)
        .arg("-g")
        .args(&args.gwas_results)
        .arg("-o")
        .arg(&args.output_file)
        .arg("--num-covar")
        .arg(args.num_covar.to_string())
        .arg("--chunksize")
        .arg(args.chunksize.to_string())
        .arg("--variant-id")
        .arg(args.variant_id)
        .arg("--beta")
        .arg(args.beta)
        .arg("--std-error")
        .arg(args.std_error)
        .arg("--sample-size")
        .arg(args.sample_size)
        .arg("--quiet");

    cmd.assert().success();

    utils::check_results(
        path.join("igwas_results.csv").to_str().unwrap(),
        path.join("direct_results.csv").to_str().unwrap(),
    );
}
