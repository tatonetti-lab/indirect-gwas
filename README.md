# Indirect GWAS

[![Project workflow](https://github.com/zietzm/indirect_GWAS/actions/workflows/python-package.yml/badge.svg)](https://github.com/zietzm/indirect_GWAS/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Indirect GWAS allows you to compute GWAS summary statistics on a linear combination of traits without running the GWAS directly and without using individual-level data.

For example, you may have GWAS summary statistics for ICD10-CM codes but want to compute a GWAS on a combination of two or more of these codes.
This package enables that analysis without needing to run any additional GWAS.

## Inputs required

Given a set of n traits $x_1$, ..., $x_n$,  define a linear combination, $z$, as the sum of $x_i \times b_i$, where $b_i$ are the user-defined projection coefficients.
To compute GWAS summary statistics on $z$, indirect GWAS requires the following data:

1. Projection coefficients $b_1$, ..., $b_n$
2. GWAS coefficients for each trait $x_1$, ..., $x_n$
3. Sample size for the indirect GWAS

and one of the following pairs:

4. Phenotypic covariances (covariate-adjusted)
5. GWAS coefficient standard errors for each trait $x_1$, ..., $x_n$

or

4. Residualized phenotypes (i.e. covariate-adjusted)
5. GWAS coefficient standard errors for each trait $x_1$, ..., $x_n$

or (for linear model GWAS only)

4. Individual-level phenotypes
5. Individual-level covariances


This library provides utilities to format data appropriately and perform this computation across a large number of projections, features, and variants simultaneously.
While no individual level data are required for indirect GWAS, we provide some simple utilities to compute covariate-adjusted covariance matrices for ease when individual data are available.

## Illustration of the method

For simplicity we'll use the `mtcars` dataset here.
Suppose that we are interested in the projected phenotype `hp - mpg`, with `vs` as the genotype variable and `gear` as an adjusted covariate.

The direct regression equivalent of this is the following:

```python
import statsmodels.api as sm

# Load data
mtcars = sm.datasets.get_rdataset("mtcars", "datasets").data

# Perform the direct regression
projection = mtcars["hp"] - mtcars["mpg"]
X = mtcars[["vs", "gear"]].assign(const=1.)
direct_reg = sm.OLS(projection, X).fit()

# Print summary information
direct_summary = pd.DataFrame({
    "coef": direct_reg.params, "se": direct_reg.bse,
    "t_stat": direct_reg.tvalues, "p": direct_reg.pvalues
}).loc["vs"]
```

Returns

|        |             vs |
|:-------|---------------:|
| coef   | -106.103       |
| se     |   18.8541      |
| t_stat |   -5.62757     |
| p      |    4.44445e-06 |


The indirect version of this regression uses only `hp` and `mpg` summary information.

```python
# Perform feature regressions first
feature_regs = {y: sm.OLS(mtcars[y], X).fit() for y in ["hp", "mpg"]}
feature_coef = pd.DataFrame({y: [reg.params["vs"]] for y, reg in regs.items()})
feature_stderr = pd.DataFrame({y: [reg.bse["vs"]] for y, reg in regs.items()})
# Partial covariance is after residualizing just the covariates
feature_partial_cov = pd.DataFrame({
    y: sm.OLS(mtcars[y], X[["gear", "const"]]).fit().resid
    for y, reg in regs.items()
}).cov()

# Setup the indirect GWAS dataset
ds = from_summary_statistics(
    projection_coefficients=pd.DataFrame([[1], [-1]], index=["hp", "mpg"]),
    feature_partial_covariance=feature_partial_cov,
    feature_gwas_coefficients=feature_coef,
    feature_gwas_standard_error=feature_stderr,
    feature_gwas_sample_size=mtcars.shape[0],
    n_covar=2,
)

# Perform indirect GWAS on the projection
beta, se, t_stat, p = gwas_indirect(ds)

# Print summary information
indirect_summary = pd.Series({
    "coef": beta.item(), "se": se.item(), "t_stat": t_stat.item(), "p": p.item()})
```

Returns

|        |             vs |
|:-------|---------------:|
| coef   | -106.103       |
| se     |   18.8541      |
| t_stat |   -5.62757     |
| p      |    4.44445e-06 |

These results are identical to 12 decimal places.

```python
assert (direct_summary - indirect_summary).abs().max() < 1e-12
```

# Setup

To install, run
```
pip install git+https://github.com/zietzm/indirect_GWAS.git
```

# FAQ

### 1. Does indirect GWAS work for linear mixed models?

Yes!
Though it depends a bit on the specific LMM in question and the data available.

* SAIGE - Reports residualized phenotypes that have been adjusted for both fixed and random effects. For more info on indirect SAIGE see the wiki.
<!-- * Regenie -->

### 2. Does indirect GWAS for for generalized linear (mixed) models?

Not exactly.
While linear and linear mixed models can be run indirectly to very high precision, generalized linear and generalized linear mixed models cannot.
This is because unlike linear and linear mixed models, generalized models do not have exact solutions and do not have an equivalent concept of a residual.
Nonetheless, linear and linear mixed models can be sufficient and good approximations in many cases.
For more on this topic, please see the [SumHer paper](https://doi.org/10.1038/s41588-018-0279-5), especially Supplementary Figure 17.

### 3. What phenotypic covariances are needed?

Indirect GWAS relies on phenotypic **partial** covariances, meaning covariances on the residual phenotypes after covariates have been removed.
In short, if every feature GWAS is run as p ~ g + c1 + ... + cm, regress each feature p ~ c1 + ... + cm, then take the regression residuals, q, and compute their covariance.

### 4. Why use this over direct GWAS?

Running GWAS on a large number of linear combinations of feature traits requires as many GWAS to be run as you define linear combinations of traits.
This method only requires as many GWAS as you have feature traits, and computes indirect GWAS in a very short amount of time (hundreds of GWAS per second).
When summary statistic data are available already (for example, from the [Neale lab GWAS](http://www.nealelab.is/uk-biobank)), indirect GWAS allows you to compute identical GWAS summary statistics for linear combinations of traits without actually having to run a single GWAS yourself.
