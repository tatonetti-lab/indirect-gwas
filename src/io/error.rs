use std::fmt;

#[derive(Debug, Clone)]
pub struct ColumnNotFound(Box<String>);

impl ColumnNotFound {
    pub fn new(column: &str) -> Self {
        ColumnNotFound(Box::new(column.to_string()))
    }
}

#[derive(Debug)]
pub enum IgwasError {
    ColumnNotFound(ColumnNotFound),
    CsvError(csv::Error),
}

impl std::error::Error for IgwasError {}

impl fmt::Display for IgwasError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            IgwasError::ColumnNotFound(ref e) => {
                write!(f, "Column: '{}' not found in header", e.0)
            }
            IgwasError::CsvError(ref e) => write!(f, "CSV error: {}", e),
        }
    }
}

impl fmt::Display for ColumnNotFound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Column: '{}' not found in header", self.0)
    }
}

impl From<csv::Error> for IgwasError {
    fn from(e: csv::Error) -> Self {
        IgwasError::CsvError(e)
    }
}

impl From<ColumnNotFound> for IgwasError {
    fn from(e: ColumnNotFound) -> IgwasError {
        IgwasError::ColumnNotFound(e)
    }
}
