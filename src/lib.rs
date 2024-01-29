use anyhow::Result;
use clap::Parser;
use log::info;
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

    /// Number of threads to use
    #[arg(short, long, default_value_t = 1)]
    pub num_threads: usize,

    /// Number of GWAS results chunks to hold in memory at once
    #[arg(long, default_value_t = 25)]
    pub capacity: usize,

    /// Whether to compress the output using zstd
    #[arg(long, default_value_t = false)]
    pub compress: bool,

    /// Suppress output
    #[arg(short, long)]
    pub quiet: bool,
}

pub fn run_cli(args: InputArguments) -> Result<()> {
    info!("Received arguments: {:#?}", &args);

    info!("Starting Indirect GWAS");
    let start = std::time::Instant::now();

    let column_names = io::gwas::ColumnSpec {
        variant_id: args.variant_id,
        beta: args.beta,
        se: args.std_error,
        sample_size: args.sample_size,
    };

    let runtime_config = util::RuntimeConfig {
        num_threads: args.num_threads,
        chunksize: args.chunksize,
        compress: args.compress,
        capacity: args.capacity,
    };

    let _pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build()
        .unwrap();

    util::run(
        &args.projection_matrix,
        &args.covariance_matrix,
        &args.gwas_results,
        &args.output_file,
        args.num_covar,
        runtime_config,
        column_names,
    )?;

    let duration = Duration::new(start.elapsed().as_secs(), 0);
    info!(
        "Finished Indirect GWAS in {}",
        humantime::format_duration(duration).to_string()
    );

    Ok(())
}
