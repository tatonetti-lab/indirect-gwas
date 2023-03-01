import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr


def gwas_indirect(data: xr.Dataset):
    """
    Compute the indirect GWAS summary statistics

    Parameters
    ----------
    dataset : xarray.Dataset
        This should have the following dimensions:
            1. "feature"
            2. "variant"
            3. "projection"
            4. "feature2"
        This should have the following data variables (dims):
            1. T (feature, projection) - projection coefficients
            2. P (feature, feature2) - feature covariance matrix
            3. s (variant,) - genotype dosage variance
            4. B (variant, feature) - GWAS coefficient estimates for features
            5. N (variant, projection) - sample size for each indirect GWAS
        You can create this array from pandas/numpy using io.build_xarray

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
    BETA_indirect = xr.dot(data["B"], data["T"], dims=["feature"])
    SE_indirect = np.sqrt(
        (
            data["T"]
            .dot(
                data["P"]
                - data["B"] * data["B"].rename(feature="feature2") * data["s"],
                dims=["feature"],
            )
            .dot(data["T"].rename(feature="feature2"), dims=["feature2"])
        )
        / ((data["N"] - 2) * data["s"])
    ).transpose("variant", "projection")
    T_STAT_indirect = BETA_indirect / SE_indirect
    P_indirect = 2 * xr.apply_ufunc(
        scipy.stats.t.sf,
        np.abs(T_STAT_indirect),
        input_core_dims=[["variant", "projection"]],
        output_core_dims=[["variant", "projection"]],
        kwargs={"df": data["N"]},
    )
    return BETA_indirect, SE_indirect, T_STAT_indirect, P_indirect


def gwas_indirect_ufunc(
    projection_coef_vec: np.ndarray | pd.DataFrame,
    feature_cov_mat: np.ndarray | pd.DataFrame,
    genotype_dosage_variance: float,
    feature_beta_hat: np.ndarray | pd.DataFrame,
    n_samples: int,
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
    n_samples : int
        Number of samples for the indirect GWAS. How many samples you could include
        for the GWAS if you were performing it directly. Typically, this is something
        like the min/max of the number of samples for each feature.

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
    beta_hat_indirect = feature_beta_hat @ projection_coef_vec

    # Standard error
    inner_term = feature_cov_mat - genotype_dosage_variance * np.outer(
        feature_beta_hat, feature_beta_hat
    )
    numerator = projection_coef_vec @ inner_term @ projection_coef_vec
    denominator = (n_samples - 2) * genotype_dosage_variance
    se_indirect = np.sqrt(numerator / denominator)

    # t-statistic
    t_stat_indirect = beta_hat_indirect / se_indirect

    # p-value
    p_indirect = 2 * scipy.stats.t.sf(np.abs(t_stat_indirect), df=n_samples)
    return beta_hat_indirect, se_indirect, t_stat_indirect, p_indirect
