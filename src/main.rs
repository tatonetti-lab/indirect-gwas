use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    let args = igwas::InputArguments::parse();
    igwas::run_cli(args)
}
