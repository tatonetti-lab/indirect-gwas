use anyhow::Result;
use clap::Parser;
use log::LevelFilter::{Error, Info};

fn main() -> Result<()> {
    let args = igwas::InputArguments::parse();

    env_logger::Builder::from_default_env()
        .filter_level(if args.quiet { Error } else { Info })
        .format_target(false)
        .target(env_logger::Target::Stdout)
        .init();

    igwas::run_cli(args)
}
