import pandas as pd
import xarray as xr
from nptyping import DataFrame, Structure, NDArray


def build_xarray(
    projection_coefficients: NDArray | DataFrame,
    feature_covariance_matrix: NDArray | DataFrame,
    genotype_dosage_variance: NDArray | pd.Series,
    feature_GWAS_coefficients: NDArray | DataFrame,
    n_samples: NDArray | DataFrame,
) -> xr.Dataset[Structure["T: float, P: float, s: float, B: float, N: int"]]:
    """
    Build a full dataset xarray from individual arrays

    Parameters
    ----------
    projection_coefficients: NDArray | pd.DataFrame, (feature x projection)
    feature_covariance_matrix: NDArray | pd.DataFrame, (feature x feature)
    genotype_dosage_variance: NDArray | pd.Series, (variant)
    feature_GWAS_coefficients: NDArray | pd.DataFrame, (variant x feature)
    n_samples: NDArray | pd.DataFrame, (variant x projection)

    Returns
    -------
    xarray.Dataset
    """
    feature_index, projection_index, variant_index = _check_inputs(**locals())

    data = {
        "T": _normalize_array(projection_coefficients, feature_index, projection_index),
        "P": _normalize_array(
            feature_covariance_matrix, feature_index, feature_index.rename("feature2")
        ),
        "s": _normalize_array(genotype_dosage_variance, variant_index, pd.Index(["s"]))[
            "s"
        ],
        "B": _normalize_array(feature_GWAS_coefficients, variant_index, feature_index),
        "N": _normalize_array(n_samples, variant_index, projection_index),
    }
    return xr.Dataset(data)


def _check_inputs(
    projection_coefficients: NDArray | DataFrame,
    feature_covariance_matrix: NDArray | DataFrame,
    genotype_dosage_variance: NDArray | pd.Series,
    feature_GWAS_coefficients: NDArray | DataFrame,
    n_samples: NDArray | DataFrame,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    # Check shapes are correct
    n_features, n_projections = projection_coefficients.shape
    n_variants = len(genotype_dosage_variance)
    assert (n_features, n_features) == feature_covariance_matrix.shape
    assert (n_variants, n_features) == feature_GWAS_coefficients.shape
    assert (n_variants, n_projections) == n_samples.shape

    # Get IDs for features, projections
    if isinstance(projection_coefficients, pd.DataFrame):
        feature_ids = projection_coefficients.index.tolist()
        projection_ids = projection_coefficients.columns.tolist()
    else:
        feature_ids = _create_names(n=n_features, prefix="F")
        projection_ids = _create_names(n=n_projections, prefix="P")

    # Get IDs for variants
    if isinstance(genotype_dosage_variance, pd.DataFrame):
        variant_ids = genotype_dosage_variance.index.tolist()
    else:
        variant_ids = _create_names(n=n_variants, prefix="V")

    # Check that index/column ids are shared where appropriate
    if isinstance(feature_covariance_matrix, pd.DataFrame):
        assert set(feature_ids) == set(feature_covariance_matrix.index)
        assert set(feature_ids) == set(feature_covariance_matrix.columns)
    if isinstance(feature_GWAS_coefficients, pd.DataFrame):
        assert set(variant_ids) == set(feature_GWAS_coefficients.index)
        assert set(feature_ids) == set(feature_GWAS_coefficients.columns)
    if isinstance(n_samples, pd.DataFrame):
        assert set(variant_ids) == set(n_samples.index)
        assert set(projection_ids) == set(n_samples.columns)

    # Create pandas indexes for each dimension
    feature_index = pd.Index(feature_ids, name="feature")
    projection_index = pd.Index(projection_ids, name="projection")
    variant_index = pd.Index(variant_ids, name="variant")
    return feature_index, projection_index, variant_index


def _normalize_array(array, row_index: pd.Index, col_index: pd.Index) -> pd.DataFrame:
    if isinstance(array, pd.DataFrame):
        return (
            array.rename_axis(row_index.name, axis="index")
            .rename_axis(col_index.name, axis="columns")
            .loc[row_index, col_index]
        )
    else:
        return pd.DataFrame(array, index=row_index, columns=col_index)


def _create_names(n, prefix):
    return [f"{prefix}{i}" for i in range(1, n + 1)]
