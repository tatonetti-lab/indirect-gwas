import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import xarray as xr
from numbers import Real


def build_xarray(
        projection_coef: ArrayLike,
        feature_cov_matrix: ArrayLike,
        genotype_dosage_variance: ArrayLike | pd.Series,
        feature_GWAS_coef: ArrayLike,
        n_samples: ArrayLike,
) -> xr.Dataset:
    """
    Build a full dataset xarray from individual arrays

    Parameters
    ----------
    projection_coef: ArrayLike, (feature x projection)
    feature_cov_matrix: ArrayLike, (feature x feature)
    genotype_dosage_variance: ArrayLike | pd.Series, (variant)
    feature_GWAS_coef: ArrayLike, (variant x feature)
    n_samples: ArrayLike, (variant x projection)

    Returns
    -------
    xarray.Dataset
        T: float, P: float, s: float, B: float, N: int
        Same order and shapes as inputs.
    """
    feature_idx, projection_idx, variant_idx = _check_inputs(**locals())

    data = {
        "T": _normalize_array(projection_coef, feature_idx, projection_idx, "T"),
        "P": _normalize_array(feature_cov_matrix, feature_idx,
                              feature_idx.rename("feature2"), "P"),
        "s": _normalize_array(genotype_dosage_variance, variant_idx, None, "s"),
        "B": _normalize_array(feature_GWAS_coef, variant_idx, feature_idx, "B"),
        "N": _normalize_array(n_samples, None, None, "N"),
    }
    return xr.Dataset(data)


def _check_inputs(
        projection_coef: ArrayLike,
        feature_cov_matrix: ArrayLike,
        genotype_dosage_variance: ArrayLike | pd.Series,
        feature_GWAS_coef: ArrayLike,
        n_samples: ArrayLike,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    # Check shapes are correct
    n_features, n_projections = projection_coef.shape[:2]
    n_variants = genotype_dosage_variance.shape[0]
    assert (n_features, n_features) == feature_cov_matrix.shape[:2]
    assert (n_variants, n_features) == feature_GWAS_coef.shape[:2]

    # Get IDs for features, projections
    if isinstance(projection_coef, pd.DataFrame):
        feature_ids = projection_coef.index.tolist()
        projection_ids = projection_coef.columns.tolist()
    elif isinstance(projection_coef, xr.DataArray):
        feature_ids = projection_coef.indexes[projection_coef.dims[0]]
        projection_ids = projection_coef.indexes[projection_coef.dims[1]]
    else:
        feature_ids = _create_names(n=n_features, prefix="F")
        projection_ids = _create_names(n=n_projections, prefix="P")

    # Get IDs for variants
    if isinstance(genotype_dosage_variance, pd.DataFrame | pd.Series):
        variant_ids = genotype_dosage_variance.index.tolist()
    elif isinstance(genotype_dosage_variance, xr.DataArray):
        variant_ids = genotype_dosage_variance.indexes[genotype_dosage_variance.dims[0]]
    else:
        variant_ids = _create_names(n=n_variants, prefix="V")

    # Check that index/column ids are shared where appropriate
    if isinstance(feature_cov_matrix, pd.DataFrame):
        assert set(feature_ids) == set(feature_cov_matrix.index)
        assert set(feature_ids) == set(feature_cov_matrix.columns)
    elif isinstance(feature_cov_matrix, xr.DataArray):
        assert set(feature_ids) == set(feature_cov_matrix.indexes[feature_cov_matrix.dims[0]])
        assert set(feature_ids) == set(feature_cov_matrix.indexes[feature_cov_matrix.dims[1]])

    if isinstance(feature_GWAS_coef, pd.DataFrame):
        assert set(variant_ids) == set(feature_GWAS_coef.index)
        assert set(feature_ids) == set(feature_GWAS_coef.columns)
    elif isinstance(feature_GWAS_coef, xr.DataArray):
        assert set(variant_ids) == set(feature_GWAS_coef.indexes[feature_GWAS_coef.dims[0]])
        assert set(feature_ids) == set(feature_GWAS_coef.indexes[feature_GWAS_coef.dims[1]])

    # Create pandas indexes for each dimension
    feature_index = pd.Index(feature_ids, name="feature")
    projection_index = pd.Index(projection_ids, name="projection")
    variant_index = pd.Index(variant_ids, name="variant")
    return feature_index, projection_index, variant_index


def _normalize_array(
        array: np.ndarray | pd.Series | pd.DataFrame | xr.DataArray,
        row_index: pd.Index | None = None,
        col_index: pd.Index | None = None,
        name: str | None = None,
) -> xr.DataArray:
    if row_index is not None:
        coords = {row_index.name: row_index}
        if col_index is not None:
            coords[col_index.name] = col_index
    else:
        coords = None

    try:
        assert isinstance(array,
                          np.ndarray | pd.Series | pd.DataFrame | xr.DataArray | Real)
    except AssertionError:
        raise AssertionError(array, type(array))

    if isinstance(array, Real):
        array = np.array(array)

    if isinstance(array, np.ndarray):
        # Doesn't support more than 2 dimensional arrays since there won't be any index
        assert name is not None

        squeezed = array.squeeze()

        if squeezed.ndim == 0:
            if row_index is None:
                # Only case where we don't need a row_index, at least
                return xr.DataArray(squeezed, name=name)
            elif col_index is None:
                return xr.DataArray(array.ravel(), coords=coords, name=name)
            else:
                return xr.DataArray(array, coords=coords, name=name)

        assert row_index is not None

        if squeezed.ndim == 1:
            if col_index is None:
                return xr.DataArray(array.ravel(), coords=coords, name=name)
            else:
                return xr.DataArray(array, coords=coords, name=name)

        assert col_index is not None
        assert squeezed.ndim >= 2

        return xr.DataArray(squeezed, coords=coords, name=name)

    elif isinstance(array, pd.Series):
        if row_index is not None:
            array = array.rename_axis(row_index.name, axis="index").loc[row_index]

        assert col_index is None
        return xr.DataArray(array, name=name)

    elif isinstance(array, pd.DataFrame):
        if row_index is not None:
            array = array.rename_axis(row_index.name, axis="index").loc[row_index]
        if col_index is not None:
            array = array.rename_axis(col_index.name, axis="columns").loc[:, col_index]

        return xr.DataArray(array, name=name)

    elif isinstance(array, xr.DataArray):
        if array.name != name:
            array = array.rename(name)

        if row_index is not None:
            array = array.loc[row_index]

        if col_index is not None:
            array = array.loc[:, col_index]

        return array
    else:
        raise ValueError(f"Unknown type of array '{type(array)}'")


def _create_names(n, prefix):
    return [f"{prefix}{i}" for i in range(1, n + 1)]
