use anyhow::Result;
use clap::Parser;
use igwas;

fn main() -> Result<()> {
    let args = igwas::InputArguments::parse();
    igwas::run_cli(args)
}
