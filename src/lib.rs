use anyhow::Result;
use clap::Parser;
use env_logger;
use humantime;
use log::info;
use log::LevelFilter::{Error, Info};
use std::time::Duration;

pub mod io;
pub mod stats;
pub mod util;

/// Perform an indirect GWAS.
/// See our preprint for more details:
/// biorxiv.org/content/10.1101/2023.11.20.567948v1
#[derive(Parser, Debug)]
#[command(author, version)]
pub struct InputArguments {
    /// Path to the projection matrix
    #[arg(short, long)]
    pub projection_matrix: String,

    /// Path to the covariance matrix
    #[arg(short = 'c', long)]
    pub covariance_matrix: String,

    /// Paths to the GWAS results files
    #[arg(num_args(1..), short, long)]
    pub gwas_results: Vec<String>,

    /// Path to the output file
    #[arg(short, long)]
    pub output_file: String,

    /// Number of covariates
    #[arg(long)]
    pub num_covar: usize,

    /// Number of variants to read per chunk
    #[arg(long, default_value_t = 100000)]
    pub chunksize: usize,

    /// Name of the variant ID column
    #[arg(short, long, default_value_t = String::from("ID"))]
    pub variant_id: String,

    /// Name of the beta column
    #[arg(short, long, default_value_t = String::from("BETA"))]
    pub beta: String,

    /// Name of the standard error column
    #[arg(long, default_value_t = String::from("SE"))]
    pub std_error: String,

    /// Name of the sample size column
    #[arg(long, default_value_t = String::from("OBS_CT"))]
    pub sample_size: String,

    /// Suppress output
    #[arg(short, long)]
    pub quiet: bool,
}

pub fn run_cli(args: InputArguments) -> Result<()> {
    let _builder = env_logger::Builder::from_default_env()
        .filter_level(if args.quiet { Error } else { Info })
        .format_target(false)
        .target(env_logger::Target::Stdout)
        .init();

    info!("Received arguments: {:#?}", &args);

    info!("Starting Indirect GWAS");
    let start = std::time::Instant::now();

    util::run(
        &args.projection_matrix,
        &args.covariance_matrix,
        &args.gwas_results,
        &args.output_file,
        args.num_covar,
        args.chunksize,
        args.variant_id,
        args.beta,
        args.std_error,
        args.sample_size,
    )?;

    let duration = Duration::new(start.elapsed().as_secs(), 0);
    info!(
        "Finished Indirect GWAS in {}",
        humantime::format_duration(duration).to_string()
    );

    Ok(())
}
