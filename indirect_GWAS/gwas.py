import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr

from nptyping import NDArray, Shape, Float


def gwas_indirect_ufunc(
    projection_coef_vec: NDArray[Shape["Features"], Float],
    feature_cov_mat: NDArray[Shape["Features, Features"], Float],
    genotype_dosage_variance: float,
    feature_beta_hat: NDArray[Shape["Features"], Float],
    n_samples: int,
) -> tuple[float, float, float, float]:
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


def gwas_indirect_xarray(dataset: xr.Dataset):
    """
    Compute the indirect GWAS summary statistics

    Parameters
    ----------
    dataset : xarray.Dataset
        This should have the following data variables:
            1. T (feature x projection) - projection coefficients
            2. P (feature x feature2) - feature covariance matrix
            3. s (variant) - genotype dosage variance
            4. B (variant x feature) - GWAS coefficient estimates for features
            5. N (variant x projection) - sample size for each indirect GWAS
        You can create this array from pandas/numpy using io.build_xarray

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]
    """
    BETA_indirect = xr.dot(dataset["B"], dataset["T"], dims=["feature"])
    SE_indirect = np.sqrt(
        (
            dataset["T"]
            .dot(
                dataset["P"]
                - dataset["B"] * dataset["B"].rename(feature="feature2") * dataset["s"],
                dims=["feature"],
            )
            .dot(dataset["T"].rename(feature="feature2"), dims=["feature2"])
        )
        / ((dataset["N"] - 2) * dataset["s"])
    )
    T_STAT_indirect = BETA_indirect / SE_indirect
    P_indirect = 2 * xr.apply_ufunc(
        scipy.stats.t.sf,
        np.abs(T_STAT_indirect),
        input_core_dims=[["variant", "projection"]],
        output_core_dims=[["variant", "projection"]],
        kwargs={"df": dataset["N"]},
    )
    return BETA_indirect, SE_indirect, T_STAT_indirect, P_indirect
