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

Download one of the [pre-compiled binaries from GitHub](https://github.com/zietzm/igwas_rs/releases/latest)

Alternatively, install from source code.
If cargo is not installed, see [cargo installation](https://doc.rust-lang.org/cargo/getting-started/installation.html).

```bash
git clone --depth 1 https://github.com/zietzm/igwas.git

cargo install --path igwas
```

## Usage

Indirect GWAS is a command line tool.

As an example,

```bash
igwas \
    -p projection.tsv \
    -c covariance.tsv \
    -g plink*.glm.linear \
    -o indirect_results.csv
```

To see a full list of parameters, run

```bash
igwas -h
```

Indirect GWAS takes four main arguments:

1. Projection matrix
2. Covariance matrix
3. GWAS result files
4. Output path

Each of these is a path in the filesystem.

### Projection matrix

This should be a CSV/TSV file with row and column names.
The first column of the first row is ignored.
For example:

```
rowid,proj1,proj2
feat1,0.1,0.2
feat2,0.2,-0.511119
```

The contents of this file should give the coefficients needed to project feature phenotypes onto the projected phenotypes.
In the example above, `proj1` is a projection defined as `0.1 * feat1 + 0.2 * feat2`.
Many projections can be passed simultaneously in this file.

### Covariance matrix

This should be a CSV/TSV file with row and column names.
The first column of the first row is ignored.
The row and column names should match, otherwise.
For example:

```
_,feat1,feat2
feat1,0.1,0.1
feat2,0.1,0.5
```

The contents of this file should give the partial covariances of the feature phenotype.
Partial covariance is defined as the covariance of the residuals of the phenotypes when regressed against the GWAS covariates.
For example, if each GWAS regression takes the form `phenotype ~ genotype + covar_1 + covar_2`, you should regress `phenotype ~ covar_1 + covar_2`, compute the residuals, do this for every phenotype, then compute the covariance matrix of these residuals.

### GWAS results

GWAS results should be formatted as CSV/TSV files.
These files should contain, at minimum, columns with the following pieces of information: variant ID, coefficient estimate, standard error, and sample size.
The column names may be specified with additional flags (e.g. `--variant-id`, `--beta`, etc.).
The default field names correspond to the outputs of Plink linear regressions.

### Output path

This should be a simple path to a single file.
This file will contains GWAS summary statistics for all the projected phenotypes, combined.
