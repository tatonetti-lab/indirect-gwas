use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader};

use anyhow::{Context, Result};
use nalgebra::{DMatrix, DVector};

pub fn count_lines(filename: &str) -> Result<usize> {
    let file = File::open(filename)?;
    let mut reader = BufReader::with_capacity(32768, file);
    let mut num_lines = 0;
    let mut string = String::new();
    while reader.read_line(&mut string)? > 0 {
        num_lines += 1;
    }
    Ok(num_lines - 1)
}

// Create a struct ColumnSpec to hold the names of columns
#[derive(Debug, Clone)]
pub struct ColumnSpec {
    pub variant_id: String,
    pub beta: String,
    pub se: String,
    pub sample_size: String,
}

struct MappedColumns {
    variant_id: usize,
    beta: usize,
    se: usize,
    sample_size: usize,
}

pub struct GwasResults {
    pub variant_ids: Vec<String>,
    pub beta_values: DVector<f32>,
    pub se_values: DVector<f32>,
    pub sample_sizes: DVector<i32>,
}

pub struct IntermediateResults {
    pub variant_ids: Vec<String>,
    pub beta_update: DMatrix<f32>,
    pub gpv_update: DVector<f32>,
    pub sample_sizes: DVector<i32>,
}

pub struct IGwasResults {
    pub projection_ids: Vec<String>,
    pub variant_ids: Vec<String>,
    pub beta_values: DVector<f32>,
    pub se_values: DVector<f32>,
    pub t_stat_values: DVector<f32>,
    pub p_values: DVector<f32>,
    pub sample_sizes: DVector<i32>,
}

fn map_column_names(header: &csv::StringRecord, spec: &ColumnSpec) -> Result<MappedColumns> {
    // Find the indices of the columns we want. If any of them are not found, return an error,
    // specifying which column was not found.
    Ok(MappedColumns {
        variant_id: header
            .iter()
            .position(|x| x == spec.variant_id)
            .context("Variant ID column not found")?,
        beta: header
            .iter()
            .position(|x| x == spec.beta)
            .context("Beta column not found")?,
        se: header
            .iter()
            .position(|x| x == spec.se)
            .context("Standard error column not found")?,
        sample_size: header
            .iter()
            .position(|x| x == spec.sample_size)
            .context("Sample size column not found")?,
    })
}

fn read_from_record<T: std::str::FromStr>(record: &csv::StringRecord, index: usize) -> T
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    // TODO: Error handling here.
    record.get(index).unwrap().parse::<T>().unwrap()
}

/// Read GWAS summary statistics from a file
/// Only reads the variant id, beta, and standard error columns
pub fn read_gwas_results(
    filename: &str,
    column_names: &ColumnSpec,
    start_line: usize,
    end_line: usize,
) -> Result<GwasResults> {
    let mut reader = csv_sniffer::Sniffer::new().open_path(filename)?;

    read_gwas_rows(&mut reader, column_names, start_line, end_line)
}

fn read_gwas_rows<T: std::io::Read>(
    reader: &mut csv::Reader<T>,
    column_names: &ColumnSpec,
    start_line: usize,
    end_line: usize,
) -> Result<GwasResults> {
    // Get the indices of the columns we want
    let header = reader.headers()?;
    let mapped_columns = map_column_names(header, column_names)?;

    let mut variant_ids: Vec<String> = Vec::new();
    let mut beta_values: Vec<f32> = Vec::new();
    let mut se_values: Vec<f32> = Vec::new();
    let mut sample_sizes: Vec<i32> = Vec::new();

    for (i, result) in reader.records().enumerate() {
        let record = result?;
        if i < start_line {
            continue;
        }
        if i >= end_line {
            break;
        }
        variant_ids.push(read_from_record(&record, mapped_columns.variant_id));
        beta_values.push(read_from_record(&record, mapped_columns.beta));
        se_values.push(read_from_record(&record, mapped_columns.se));
        sample_sizes.push(read_from_record(&record, mapped_columns.sample_size));
    }

    // Return the results
    Ok(GwasResults {
        variant_ids,
        beta_values: DVector::from_vec(beta_values),
        se_values: DVector::from_vec(se_values),
        sample_sizes: DVector::from_vec(sample_sizes),
    })
}

fn write_rows<W: std::io::Write>(
    writer: &mut csv::Writer<W>,
    results: &IGwasResults,
    add_header: bool,
) -> Result<()> {
    if add_header {
        writer.write_record([
            "phenotype_id",
            "variant_id",
            "beta",
            "std_error",
            "t_stat",
            "p_value",
            "sample_size",
        ])?;
    }

    for i in 0..results.variant_ids.len() {
        writer.write_record(&[
            results.projection_ids[i].clone(),
            results.variant_ids[i].clone(),
            results.beta_values[i].to_string(),
            results.se_values[i].to_string(),
            results.t_stat_values[i].to_string(),
            results.p_values[i].to_string(),
            results.sample_sizes[i].to_string(),
        ])?;
    }

    Ok(())
}

pub fn write_gwas_results(
    results: IGwasResults,
    filename: &str,
    add_header: bool,
    compress: bool,
) -> Result<()> {
    let file = if add_header {
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(filename)?
    } else {
        OpenOptions::new().append(true).open(filename)?
    };

    if compress {
        let zstd_writer = zstd::stream::write::Encoder::new(file, 0)?.auto_finish();
        let mut writer = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .buffer_capacity(8 * (1 << 13))
            .from_writer(zstd_writer);
        write_rows(&mut writer, &results, add_header)?;
    } else {
        let mut writer = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .buffer_capacity(8 * (1 << 13))
            .from_writer(file);
        write_rows(&mut writer, &results, add_header)?;
    };

    Ok(())
}
