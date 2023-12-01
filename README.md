# Indirect GWAS

Indirect GWAS is a Rust program for computing genome-wide association study results indirectly.
Unlike traditional methods, indirect GWAS generates GWAS summary statistics for a phenotype definition using only other summary statistics.
To do so, we approximate a target phenotype using phenotypes for which GWAS summary statistics are already available.

As an example, indirect GWAS allows you to compute GWAS summary statistics for phecodes using only summary statistics about ICD-10 codes.

Traditional approach:
1. Define phenotype in terms of clinically-observed features
2. Evaluate phenotype for every individual
3. Perform GWAS

Indirect approach:
1. Define phenotype in terms of features that have available GWAS summary statistics (using e.g. Pan-UKBB summary statistics)
2. Compute GWAS summary statistics for the target using feature summary statistics as inputs

## Installation

Download one of the pre-compiled binaries from [GitHub](https://github.com/zietzm/igwas_rs/releases/latest)
