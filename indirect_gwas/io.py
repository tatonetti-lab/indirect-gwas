import numpy as np
import pandas as pd
from xarray import Dataset


class IndirectGWASDataset(Dataset):
    __slots__ = ("T", "P", "s", "B", "df")


def compute_phenotypic_partial_covariance(
    feature_phenotypes: pd.DataFrame,
    covariates: pd.DataFrame,
    add_intercept: bool = True,
):
    """
    Compute the partial covariance of phenotypes not explained by covariates using
    individual-level data.

    This is the covariance of the residuals that arise when covariates are used to
    predict each feature phenotype. Only run this if needed. Runtime can be minutes or
    more.

    Parameters
    ----------
    feature_phenotypes : pd.DataFrame
        samples x features
    covariates : pd.DataFrame
        samples x covariates
    add_intercept : bool, optional
        Whether to add an intercept/constant term to the covariates, by default True

    Returns
    -------
    pd.DataFrame
        Covariance among feature phenotypes after adjusting for covariates
    """
    # Add an intercept if there is not one
    if add_intercept:
        covariates = covariates.copy()
        covariates["const"] = 1

    # Residualize phenotypes against covariates
    residualization_coef = np.linalg.lstsq(covariates, feature_phenotypes, None)[0]
    predictions = (covariates @ residualization_coef).set_axis(
        feature_phenotypes.columns, axis=1
    )
    residualized_phenotypes = feature_phenotypes - predictions
    phenotype_partial_covariance = np.cov(residualized_phenotypes, rowvar=False, ddof=1)
    if phenotype_partial_covariance.ndim < 2:
        phenotype_partial_covariance = phenotype_partial_covariance.reshape(1, -1)

    # Convert to pandas.DataFrame with proper index and columns
    phenotype_partial_covariance = pd.DataFrame(
        phenotype_partial_covariance,
        index=feature_phenotypes.columns,
        columns=feature_phenotypes.columns,
    )
    return phenotype_partial_covariance


def compute_genotype_partial_variance(
    feature_partial_covariance: pd.DataFrame,
    feature_gwas_coefficients: pd.DataFrame,
    feature_gwas_standard_error: pd.DataFrame,
    feature_gwas_dof: int | pd.DataFrame,
) -> pd.Series:
    """
    Compute the genotype partial variance for each variant.

    Consider the GWAS as p ~ g + covar. Let `n` be the number of samples, let `n_covar`
     be the number of covariates, let `beta_g` be the estimated coefficient of g, let
     `SE(beta_g)` be the coefficient's standard error, and let `Var_p(p)` be the partial
     phenotypic variance. Then the partial genotype variance is computed as follows:

    Var_p(g) = Var_p(p) / (SE(beta_g)^2 * (n - n_covar - 1) + beta_g^2)

    Note that at present this doesn't allow projections to have different sample sizes,
    which would result in slightly different genotype partial variances.

    This function assumes that all inputs have consistent indexes.
    """
    # Compute the variants x features matrix of genotype partial variance
    feature_partial_variance = pd.Series(
        np.diag(feature_partial_covariance), index=feature_partial_covariance.index
    )
    feature_genotype_partial_variance = feature_partial_variance / (
        feature_gwas_standard_error**2 * feature_gwas_dof
        + feature_gwas_coefficients**2
    )

    # Take the mean of this array over the features to get a per-variant value
    genotype_partial_variance = np.mean(feature_genotype_partial_variance, axis=1)

    # Convert to pandas.Series with proper index and name
    return pd.Series(
        genotype_partial_variance,
        index=feature_gwas_coefficients.index,
        name="s",
    )


def _check_projection_coefficients(
    projection_coefficients: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure that projection coefficients are consistent"""
    # Check that input is a DataFrame
    assert isinstance(projection_coefficients, pd.DataFrame)

    # Ensure no duplicates in indexes or columns (enumerated for debugging clarity)
    assert not projection_coefficients.index.has_duplicates
    assert not projection_coefficients.columns.has_duplicates

    # Check that the indices are consistent (same values)
    projection_index = projection_coefficients.columns.sort_values().rename(
        "projection"
    )
    feature_index = projection_coefficients.index.sort_values().rename("feature")

    # Normalize DataFrame (ensure consistent index/cols)
    projection_coefficients = projection_coefficients.loc[
        feature_index, projection_index
    ]

    return projection_coefficients


def _check_individual_data(
    feature_phenotypes: pd.DataFrame, covariates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure that individual data is consistent, assume index values are sample ids"""
    # Check that all inputs are DataFrames
    assert isinstance(feature_phenotypes, pd.DataFrame)
    assert isinstance(covariates, pd.DataFrame)

    # Ensure no duplicates in indexes or columns (enumerated for debugging clarity)
    assert not feature_phenotypes.index.has_duplicates
    assert not feature_phenotypes.columns.has_duplicates
    assert not covariates.index.has_duplicates
    assert not covariates.columns.has_duplicates

    # Check that the indices are consistent (same sample ids)
    feature_index = feature_phenotypes.index.sort_values()
    covariate_index = covariates.index.sort_values()
    assert feature_index.equals(covariate_index)

    # Normalize DataFrames (ensure consistent index/cols)
    feature_phenotypes = feature_phenotypes.loc[feature_index]
    covariates = covariates.loc[feature_index]

    return feature_phenotypes, covariates


def _check_gwas_sumstats(*arrays: pd.DataFrame) -> list[pd.DataFrame]:
    """Ensure that coefficients, standard errors, and sample sizes are consistent"""
    # Check that all inputs are DataFrames
    for array in arrays:
        assert isinstance(array, pd.DataFrame)

    # Check that all inputs have the same shape
    assert len(set(array.shape for array in arrays)) == 1

    # Ensure no duplicates in indexes or columns
    for array in arrays:
        assert not array.index.has_duplicates
        assert not array.columns.has_duplicates

    # Check that the arbitrary number of dataframes have identical index and columns
    row_index = arrays[0].index.sort_values().rename("variant")
    col_index = arrays[0].columns.sort_values().rename("feature")

    for array in arrays:
        assert array.index.sort_values().equals(row_index)
        assert array.columns.sort_values().equals(col_index)

    # Normalize each DataFrame (ensure consistent index/cols)
    arrays = [array.loc[row_index, col_index] for array in arrays]

    return arrays


def _check_feature_partial_covariance(
    feature_partial_covariance: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure that feature partial covariance is consistent"""
    # Check that input is a DataFrame
    assert isinstance(feature_partial_covariance, pd.DataFrame)

    # Ensure no duplicates in indexes or columns (enumerated for debugging clarity)
    assert not feature_partial_covariance.index.has_duplicates
    assert not feature_partial_covariance.columns.has_duplicates

    # Check that the indices are consistent (same values)
    feature_index = feature_partial_covariance.columns.sort_values().rename("feature")
    assert feature_index.equals(feature_partial_covariance.index.sort_values())

    # Normalize DataFrame (ensure consistent index/cols)
    feature_partial_covariance = feature_partial_covariance.loc[
        feature_index, feature_index.rename("feature2")
    ]

    return feature_partial_covariance


def _check_genotype_partial_variance(genotype_partial_variance: pd.Series) -> pd.Series:
    """Ensure that genotype partial variance is consistent"""
    # Check that input is a Series
    assert isinstance(genotype_partial_variance, pd.Series)

    # Ensure no duplicates in index (enumerated for debugging clarity)
    assert not genotype_partial_variance.index.has_duplicates

    # Format the index
    variant_index = genotype_partial_variance.index.sort_values().rename("variant")

    return genotype_partial_variance.loc[variant_index].rename("s")


def _check_projection_degrees_of_freedom(
    projection_degrees_of_freedom: int | pd.Series,
) -> int | pd.Series:
    """Ensure that projection degrees of freedom is consistent.
    Either an integer or a pd.Series across variants."""
    # Check that input is a Series or int
    assert isinstance(projection_degrees_of_freedom, pd.Series | int)

    # Ensure no duplicates in index (enumerated for debugging clarity)
    if isinstance(projection_degrees_of_freedom, pd.Series):
        assert not projection_degrees_of_freedom.index.has_duplicates

        # Format the index
        variant_index = projection_degrees_of_freedom.index.sort_values().rename(
            "variant"
        )
        return projection_degrees_of_freedom.loc[variant_index].rename("N")
    else:
        assert projection_degrees_of_freedom > 0
        return projection_degrees_of_freedom


def from_individual_data(
    projection_coefficients: pd.DataFrame,
    feature_phenotypes: pd.DataFrame,
    covariates: pd.DataFrame,
    feature_gwas_coefficients: pd.DataFrame,
    feature_gwas_standard_error: pd.DataFrame,
    feature_gwas_sample_size: int | pd.DataFrame,
    add_intercept: bool = True,
) -> IndirectGWASDataset:
    """
    Build an IndirectGWASDataset from individual-level data.

    Computes the phenotypic partial covariance matrix and the genotype partial variance.
    If you just finished running a linear regression GWAS, this is likely the simplest
    method to use.

    Note: This method will only work for linear regression GWAS. If you have a linear
    mixed model or any covariates that aren't provided, you'll need to compute the
    phenotypic partial covariance matrix first.

    This function is essentially just a wrapper around
    `compute_phenotypic_partial_variance` and `from_summary_statistics`.

    Parameters
    ----------
    projection_coefficients : pd.DataFrame
        features x projections
    feature_phenotypes : pd.DataFrame
        samples x features
    covariates : pd.DataFrame
        samples x covariates
    feature_gwas_coefficients : pd.DataFrame
        variants x features
    feature_gwas_standard_error : pd.DataFrame
        variants x features
    feature_gwas_sample_size : int | pd.DataFrame
        The number of samples used in each feature GWAS regression. Either the same
        value for every variant x feature or a DataFrame across (variants x features).
    add_intercept : bool, optional
        Whether to add a constant intercept term to the covariates, by default True

    Returns
    -------
    IndirectGWASDataset
    """
    # Check projection coefficients
    projection_coefficients = _check_projection_coefficients(projection_coefficients)

    # Check individual data
    feature_phenotypes, covariates = _check_individual_data(
        feature_phenotypes, covariates
    )

    # Check GWAS summary statistics
    (
        feature_gwas_coefficients,
        feature_gwas_standard_error,
    ) = _check_gwas_sumstats(
        feature_gwas_coefficients,
        feature_gwas_standard_error,
    )
    if isinstance(feature_gwas_sample_size, pd.DataFrame):
        [feature_gwas_sample_size] = _check_gwas_sumstats(feature_gwas_sample_size)

    # Compute the phenotypic partial covariance matrix
    feature_partial_covariance = compute_phenotypic_partial_covariance(
        feature_phenotypes,
        covariates,
        add_intercept=add_intercept,
    )
    n_covar = covariates.shape[1] if not add_intercept else covariates.shape[1] + 1
    return from_summary_statistics(
        projection_coefficients,
        feature_partial_covariance,
        feature_gwas_coefficients,
        feature_gwas_standard_error,
        feature_gwas_sample_size - n_covar - 1,
    )


def from_summary_statistics(
    projection_coefficients: pd.DataFrame,
    feature_partial_covariance: pd.DataFrame,
    feature_gwas_coefficients: pd.DataFrame,
    feature_gwas_standard_error: pd.DataFrame,
    feature_gwas_dof: int | pd.DataFrame,
) -> IndirectGWASDataset:
    """
    Build an IndirectGWASDataset from summary statistics.

    Computes the genotype partial variance using the phenotypic partial covariance
    matrix and the feature GWAS summary statistics.

    This function is essentially just a wrapper around
    `compute_genotype_partial_variance` and `from_final_data`.

    Parameters
    ----------
    projection_coefficients : pd.DataFrame
        features x projections
    feature_partial_covariance : pd.DataFrame
        features x features
    feature_gwas_coefficients : pd.DataFrame
        variants x features
    feature_gwas_standard_error : pd.DataFrame
        variants x features
    feature_gwas_dof : int | pd.DataFrame
        Degrees of freedom in each feature GWAS regression. Either the same
        value for every variant x feature or a DataFrame across (variants x features).

    Returns
    -------
    IndirectGWASDataset
    """
    # Check projection coefficients
    projection_coefficients = _check_projection_coefficients(projection_coefficients)

    # Check phenotypic partial covariance matrix
    feature_partial_covariance = _check_feature_partial_covariance(
        feature_partial_covariance
    )

    # Check GWAS summary statistics
    (
        feature_gwas_coefficients,
        feature_gwas_standard_error,
    ) = _check_gwas_sumstats(feature_gwas_coefficients, feature_gwas_standard_error)

    if isinstance(feature_gwas_dof, pd.DataFrame):
        [feature_gwas_dof] = _check_gwas_sumstats(feature_gwas_dof)

    # Compute the genotype partial variances
    genotype_partial_variance = compute_genotype_partial_variance(
        feature_partial_covariance=feature_partial_covariance,
        feature_gwas_coefficients=feature_gwas_coefficients,
        feature_gwas_standard_error=feature_gwas_standard_error,
        feature_gwas_dof=feature_gwas_dof,
    )

    # Set the projection d.o.f. to min of feature GWAS d.o.f.
    if isinstance(feature_gwas_dof, pd.DataFrame):
        projection_dof = feature_gwas_dof.min(axis=1)
    else:
        projection_dof = feature_gwas_dof

    # Build the dataset
    return from_final_data(
        projection_coefficients,
        feature_partial_covariance,
        feature_gwas_coefficients,
        genotype_partial_variance,
        projection_dof,
    )


def from_final_data(
    projection_coefficients: pd.DataFrame,
    feature_partial_covariance: pd.DataFrame,
    feature_gwas_coefficients: pd.DataFrame,
    genotype_partial_variance: pd.Series,
    projection_degrees_of_freedom: int | pd.Series,
) -> IndirectGWASDataset:
    """
    Build an IndirectGWASDataset from the final data.

    Use this function only if you already have the phenotypic partial covariance matrix
    and the genotype partial variance for every variant.

    Parameters
    ----------
    projection_coefficients : pd.DataFrame
        features x projections
    feature_partial_covariance : pd.DataFrame
        features x features
    feature_gwas_coefficients : pd.DataFrame
        variants x features
    genotype_partial_variance : pd.Series
        variants
    projection_degrees_of_freedom : int | pd.Series
        The number of degrees of freedom in each indirect GWAS regression. Either the
        same value for all variants and indirect regressions (int) or a variant-specific
        value (pd.Series). Currently does not support projection-specific values.

    Returns
    -------
    IndirectGWASDataset
    """
    # Check inputs
    # Check projection coefficients
    projection_coefficients = _check_projection_coefficients(projection_coefficients)

    # Check phenotypic partial covariance matrix
    feature_partial_covariance = _check_feature_partial_covariance(
        feature_partial_covariance
    )

    # Check GWAS summary statistics
    [feature_gwas_coefficients] = _check_gwas_sumstats(feature_gwas_coefficients)

    # Check genotype partial variance
    genotype_partial_variance = _check_genotype_partial_variance(
        genotype_partial_variance
    )

    # Check projection degrees of freedom
    projection_degrees_of_freedom = _check_projection_degrees_of_freedom(
        projection_degrees_of_freedom
    )

    # Check that indexes are consistent with one another
    assert (
        projection_coefficients.index.equals(feature_partial_covariance.index)
        and feature_partial_covariance.index.equals(feature_gwas_coefficients.columns)
        and feature_gwas_coefficients.index.equals(genotype_partial_variance.index)
    ), "Indexes are not consistent with one another"

    if isinstance(projection_degrees_of_freedom, pd.Series):
        assert projection_degrees_of_freedom.index.equals(
            genotype_partial_variance.index
        ), "Indexes are not consistent with one another"

    # Build the dataset
    return IndirectGWASDataset(
        {
            "T": projection_coefficients,
            "P": feature_partial_covariance,
            "s": genotype_partial_variance,
            "B": feature_gwas_coefficients,
            "df": projection_degrees_of_freedom,
        }
    )
