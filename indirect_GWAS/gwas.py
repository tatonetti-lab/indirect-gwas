import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr

from .io import IndirectGWASDataset


def gwas_indirect(data: IndirectGWASDataset):
    """
    Compute the indirect GWAS summary statistics

    Parameters
    ----------
    data : IndirectGWASDataset
        This should have the following dimensions:
            1. "feature"
            2. "variant"
            3. "projection"
            4. "feature2"
        This should have the following data variables (required dims first, ...=any):
            1. T (feature, projection, ...) - projection coefficients
            2. P (feature, feature2, ...) - feature covariance matrix
            3. s (variant, ...) - genotype dosage variance
            4. B (variant, feature, ...) - GWAS coefficient estimates for features
            5. df (#variant ...) - degrees of freedom for each indirect GWAS

        You can create this array, ensuring consistency using io.from_* functions

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]
        BETA, SE, T_STAT, P
        GWAS summary statistics, each with dims = (variant, projection)

    Notes
    -----
    This uses a single xarray so that we can be sure the different arrays are indexed
    and aligned in the same way.
    """
    # Coefficient estimates
    BETA_indirect = xr.dot(data["B"], data["T"], dims=["feature"])

    # Standard errors
    # Partial variance of the projected traits
    var_p_z = (
        data["T"]
        .dot(data["P"], dims=["feature"])
        .dot(data["T"].rename(feature="feature2"), dims=["feature2"])
    )
    SE_indirect = np.sqrt((var_p_z / data["s"] - BETA_indirect**2) / data["df"])

    # t-statistics
    T_STAT_indirect = BETA_indirect / SE_indirect

    # p-values
    P_indirect = 2 * xr.apply_ufunc(
        scipy.stats.t.sf,
        np.abs(T_STAT_indirect),
        data["df"],
        input_core_dims=[[], []],
        output_core_dims=[[]],
    )
    return (
        BETA_indirect.transpose("variant", "projection", ...).rename("beta"),
        SE_indirect.transpose("variant", "projection", ...).rename("se"),
        T_STAT_indirect.transpose("variant", "projection", ...).rename("t_stat"),
        P_indirect.transpose("variant", "projection", ...).rename("p"),
    )


def gwas_indirect_ufunc(
    projection_coef_vec: np.ndarray | pd.DataFrame,
    feature_cov_mat: np.ndarray | pd.DataFrame,
    genotype_dosage_variance: float,
    feature_beta_hat: np.ndarray | pd.DataFrame,
    df: int,
) -> tuple[float, float, float, float]:
    """
    Universal function to compute the indirect GWAS summary statistics for one variant
    and one projection at a time.

    Parameters
    ----------
    projection_coef_vec : np.ndarray | pd.DataFrame, shape (n_features, n_projections)
        Vector used to weight features in the projection
    feature_cov_mat : np.ndarray | pd.DataFrame, shape (n_features, n_features)
        Covariance matrix of the features
    genotype_dosage_variance : float
        Variance of the genotype/dosage/whatever you would use as the independent
        variable in GWAS
    feature_beta_hat : np.ndarray | pd.DataFrame, shape (n_features,)
        GWAS coefficients for the variant against each feature trait
    df : int
        Degrees of freedom for the indirect GWAS. How many samples you could include
        for the GWAS if you were performing it directly minus the number of covariates
        minus one.

    Returns
    -------
    tuple[float, float, float, float]
        BETA, SE, T_STAT, P
        GWAS summary statistics

    Notes
    -----
    In general, this function is much slower than `gwas_indirect_xarray`. The only
    reason to include it at all is for possible speedups during parallelization.
    """
    # Coefficient
    beta = feature_beta_hat @ projection_coef_vec

    # Standard error
    var_p_z = projection_coef_vec @ feature_cov_mat @ projection_coef_vec
    se = np.sqrt((var_p_z / genotype_dosage_variance - beta**2) / df)

    # t-statistic
    t_stat = beta / se

    # p-value
    p_value = 2 * scipy.stats.t.sf(np.abs(t_stat), df=df)
    return beta, se, t_stat, p_value
