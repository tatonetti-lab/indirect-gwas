extern crate csv;
extern crate nalgebra as na;

use na::DVector;

// Create a struct ColumnSpec to hold the names of columns
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

pub struct IGwasResults {
    pub projection_ids: Vec<String>,
    pub variant_ids: Vec<String>,
    pub beta_values: DVector<f32>,
    pub se_values: DVector<f32>,
    pub t_stat_values: DVector<f32>,
    pub p_values: DVector<f32>,
    pub sample_sizes: DVector<i32>,
}

fn map_column_names(header: &csv::StringRecord, spec: &ColumnSpec) -> MappedColumns {
    // TODO: Error handling here.
    MappedColumns {
        variant_id: header.iter().position(|x| x == spec.variant_id).unwrap(),
        beta: header.iter().position(|x| x == spec.beta).unwrap(),
        se: header.iter().position(|x| x == spec.se).unwrap(),
        sample_size: header.iter().position(|x| x == spec.sample_size).unwrap(),
    }
}

// TODO: Error handling here.
fn read_from_record<T: std::str::FromStr>(record: &csv::StringRecord, index: usize) -> T
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    record.get(index).unwrap().parse::<T>().unwrap()
}

/// Read GWAS summary statistics from a file
/// Only reads the variant id, beta, and standard error columns
pub fn read_gwas_results(
    filename: &str,
    column_names: &ColumnSpec,
    start_line: usize,
    end_line: usize,
) -> Result<GwasResults, csv::Error> {
    let mut reader = csv::Reader::from_path(filename)?;

    // Get the indices of the columns we want
    let header = reader.headers()?;
    let mapped_columns = map_column_names(&header, column_names);

    let mut variant_ids: Vec<String> = Vec::new();
    let mut beta_values: Vec<f32> = Vec::new();
    let mut se_values: Vec<f32> = Vec::new();
    let mut sample_sizes: Vec<i32> = Vec::new();

    for (i, result) in reader.records().enumerate() {
        let record = result?;
        if i >= start_line && i <= end_line {
            variant_ids.push(read_from_record(&record, mapped_columns.variant_id));
            beta_values.push(read_from_record(&record, mapped_columns.beta));
            se_values.push(read_from_record(&record, mapped_columns.se));
            sample_sizes.push(read_from_record(&record, mapped_columns.sample_size));
        }
    }

    // Return the results
    Ok(GwasResults {
        variant_ids,
        beta_values: DVector::from_vec(beta_values),
        se_values: DVector::from_vec(se_values),
        sample_sizes: DVector::from_vec(sample_sizes),
    })
}

pub fn write_gwas_results(results: IGwasResults, filename: &str) -> Result<(), csv::Error> {
    let mut writer = csv::Writer::from_path(filename)?;

    // Write the header
    writer.write_record(&[
        "phenotype",
        "variant_id",
        "beta",
        "se",
        "t_stat",
        "p_value",
        "sample_size",
    ])?;

    // Write the results
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
