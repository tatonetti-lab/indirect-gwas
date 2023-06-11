# Indirect GWAS

[![Project workflow](https://github.com/tatonetti-lab/indirect-gwas/actions/workflows/python-package.yml/badge.svg)](https://github.com/tatonetti-lab/indirect-gwas/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Indirect GWAS allows you to compute GWAS summary statistics on a linear combination of traits without running the GWAS directly and without using individual-level data.

For example, you may have GWAS summary statistics for ICD10-CM codes but want to compute a GWAS on a combination of two or more of these codes.
This package enables that analysis without needing to run any additional GWAS.

We provide both command line and Python interfaces.

## Example usage

```bash
indirect-GWAS \
  --projection-coefficients coef.csv \
  --gwas-summary-statistics gwas/*.linear \
  --variant-id-column RSID \
  --coefficient-column BETA \
  --standard-error-column SE \
  --sample-size-column N \
  --equal-number-of-covariates 12 \
  --output gwas/indirect
```

# Setup

To install, run
```
pip install git+https://github.com/tatonetti-lab/indirect-gwas.git
```

This package uses Python >=3.10 and depends on numpy, pandas, scipy, and xarray.

Once installed, this package exposes two command line arguments: `indirect-gwas` and `compute-feature-partial-covariance`.
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
We include a helper script `compute-feature-partial-covariance`, which can be run from the command line after installing the package.
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
