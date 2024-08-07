use anyhow::{Context, Result};
use nalgebra::DMatrix;

pub struct LabeledMatrix {
    pub row_labels: Vec<String>,
    pub col_labels: Vec<String>,
    pub matrix: DMatrix<f32>,
}

/// Read a matrix from a file
pub fn read_labeled_matrix(filename: &str) -> Result<LabeledMatrix> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(if filename.ends_with(".csv") {
            b','
        } else {
            b'\t'
        })
        .from_path(filename)
        .with_context(|| format!("Failed to open file {}", filename))?;

    let mut row_labels = Vec::new();
    let mut matrix = Vec::new();

    // Read the col_labels from the reader header property
    let col_labels: Vec<String> = reader
        .headers()?
        .clone()
        .iter()
        .skip(1)
        .map(|x| x.to_string())
        .collect();

    for result in reader.records() {
        let record = result?;
        for (i, value) in record.iter().enumerate() {
            if i == 0 {
                row_labels.push(value.to_string());
            } else {
                match value.parse::<f32>() {
                    Ok(v) => matrix.push(v),
                    Err(_) => matrix.push(f32::NAN),
                }
            }
        }
    }

    // Convert the matrix to a DMatrix
    let matrix = DMatrix::from_row_slice(row_labels.len(), col_labels.len(), &matrix);

    Ok(LabeledMatrix {
        row_labels,
        col_labels,
        matrix,
    })
}
