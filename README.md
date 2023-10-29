# Indirect GWAS

[![Project workflow](https://github.com/tatonetti-lab/indirect-gwas/actions/workflows/python-package.yml/badge.svg)](https://github.com/tatonetti-lab/indirect-gwas/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Indirect GWAS allows you to compute GWAS summary statistics on a linear combination of traits without running the GWAS directly and without using individual-level data.

For example, you may have GWAS summary statistics for ICD10-CM codes but want to compute a GWAS on a combination of two or more of these codes.
This package enables that analysis without needing to run any additional GWAS.

We provide both command line and Python interfaces.

## Example usage

```bash
indirect-gwas \
  --projection-coefficients coef.csv \
  --gwas-summary-statistics phenotype1.glm.linear phenotype2.glm.linear \
  --variant-id-column RSID \
  --coefficient-column BETA \
  --standard-error-column SE \
  --sample-size-column N \
  --number-of-covariates 12 \
  --output indirect
```

# Installation

We will distribute this as a Python package with binary wheels for Linux, MacOS, and Windows.
Currently, only source code is available.
See below for a source code installation.

# Installation from source

## C++ dependencies

Indirect GWAS relies on three external libraries: [Eigen](https://eigen.tuxfamily.org), [Boost.Math](https://www.boost.org/doc/libs/1_77_0/libs/math/doc/html/index.html), and [csv-parser](https://github.com/vincentlaucsb/csv-parser).
As Eigen and Boost.Math are extensive libraries, they are not included in this package and must be installed separately.
If these are already installed in a standard location, then no additional setup is required.
Otherwise, you must specify the location of these libraries using environment variables.

### 1. Libraries not already installed

A short example of how to install these libraries on a Linux system is given below.

```bash
# Move to a directory where you want to install the libraries
cd <wherever you want to install the libraries>

# Install Eigen
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
tar -xvf eigen-3.4.0.tar.bz2
export EIGEN_INCLUDE_PATH=$(realpath eigen-3.4.0)

# Install Boost.Math (latest stable version)
wget https://github.com/boostorg/math/archive/refs/heads/master.zip
unzip master.zip
mv math-master boostmath
export BOOST_MATH_INCLUDE_PATH=$(realpath boostmath)/include

# Install csv-parser (latest stable version)
mkdir csv-parser
wget -O csv-parser https://raw.githubusercontent.com/vincentlaucsb/csv-parser/master/single_include/csv.hpp
export CSV_INCLUDE_PATH=$(realpath csv-parser)
```

### 2. Libraries already installed, but not in a standard location

If these libraries were already installed but not in a standard location,
set the environment variables to the paths where you installed the libraries.

```bash
# If not set already, set the environment variables to the paths
# where you installed the libraries. If these are already installed
# in a standard location, then skip this step.
export EIGEN_INCLUDE_PATH=  # Fill in path here
export BOOST_MATH_INCLUDE_PATH=  # Fill in path here
export CSV_INCLUDE_PATH=  # Fill in path here
```

## Install Python package

By this point, Eigen, Boost.Math, and csv-parser should be installed, either in standard
locations like `/usr/include` or in custom locations specified by environment variables.

```bash
pip install git+https://github.com/tatonetti-lab/indirect-gwas.git
```

Once installed, this package exposes `indirect-gwas`.
For help with their arguments, either use the command line help (e.g. `indirect-gwas --help`) or see the wiki.

# Inputs required

Define a new projected phenotype, $z$ as

$$
z = \sum_i \beta_i x_i
$$

where $x_i$ are the feature phenotypes and $\beta_i$ are the projection coefficients.

**Indirect GWAS lets you do a GWAS on $z$ using only $\beta_i$ and summary statistics about $x_i$.**

This requires the following data:

1. User-defined projection coefficients $\beta_i$
2. GWAS summary statistics for each trait $x_i$ (coefficients, standard errors, degrees of freedom)
3. Phenotypic partial covariance matrix (i.e. covariate-adjusted covariance)

## Projection coefficients

Projections are different combinations of the $x$ variables (e.g. $z$ above is one projection).
This software can run indirect GWAS on many projections simultaneously.

This should be a delimited file (e.g. `.csv`, `.tsv`, etc.) with shape features x projections.
First column should be the feature names and first row should be the projection names (unless disabled using `--no-index`).
Specified using `-P`/`--projection-coefficients`.

## GWAS summary statistics

These should be given as one file per feature phenotype using `-gwas`/`--gwas-summary-statistics`.
Coefficients and their standard errors must be columns whose names you specify (`--coefficient-column`, `--standard-error-column`).

Degrees of freedom can be defined a number of ways.
1. A constant value for every GWAS regression (only use this if the degrees of freedom for every feature x variant is the same) (`--degrees-of-freedom`)
2. A column in the summary statistics files giving the degrees of freedom (`--degrees-of-freedom-column`)
3. Through sample size (N) and number of covariates (K) $\rightarrow$ (DF = N - K - 1).

    a. Sample size is either a constant for all (`--equal-sample-size`) or a column in the summary statistic files (`--sample-size-column`)

    b. Number of covariates is either a constant for all (`--equal-number-of-covariates`) or a column in the summary statistic files (`--number-of-covariates-column`)

The column used to identify variants must also be specified using `--variant-id-column`.

## Phenotypic partial covariance matrix

The phenotypic partial covariance matrix is the covariance matrix of feature phenotypes after residualizing covariates.
For example, suppose the GWAS model is $y \sim g + \mathrm{age} + \mathrm{sex} + \mathrm{PC1} + ... + \mathrm{PC10}$.
Let $\hat{y}$ be the fitted values from the following regression: $y \sim \mathrm{age} + \mathrm{sex} + \mathrm{PC1} + ... + \mathrm{PC10}$.
Then the residuals for feature phenotype $i$ are $r_i = y_i - \hat{y}_i$.
The phenotypic partial covariance matrix is the covariance matrix of all the feature residuals.
For linear model GWAS, these residuals can be computed quickly and easily using any regression software.
For linear mixed models such as SAIGE, this matrix can be computed using outputs from the methods themselves (see wiki for more information).

# FAQ

### 1. Does indirect GWAS work for linear mixed models?

Yes!
Though it depends a bit on the specific LMM in question and the data available.

* SAIGE - Reports residualized phenotypes that have been adjusted for both fixed and random effects. For more info on indirect SAIGE see the wiki.
<!-- * Regenie -->

### 2. Does indirect GWAS work for generalized linear models?

Not exactly.
While linear and linear mixed models can be run indirectly to very high precision, generalized linear and generalized linear mixed models cannot.
This is because unlike linear model, generalized linear models do not have exact solutions and do not have an equivalent concept of a residual.
Nonetheless, linear and linear mixed models can be sufficient and good approximations in many cases.
For more on this topic, please see the [SumHer paper](https://doi.org/10.1038/s41588-018-0279-5), especially Supplementary Figure 17.

### 3. Why use this over direct GWAS?

Running GWAS on a large number of linear combinations of feature traits requires as many GWAS to be run as you define linear combinations of traits.
This method only requires as many GWAS as you have feature traits, and computes indirect GWAS in a very short amount of time (hundreds of GWAS per second).
When summary statistic data are available already (for example, from the [Neale lab GWAS](http://www.nealelab.is/uk-biobank)), indirect GWAS allows you to compute identical GWAS summary statistics for linear combinations of traits without actually having to run a single GWAS yourself.
