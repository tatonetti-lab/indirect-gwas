import numpy as np
import pandas as pd
import pytest

from indirect_GWAS.io import _create_names, build_xarray


def build_data_numpy(n_features, n_projections, n_variants, one_dim, seed):
    np.random.seed(seed)

    feature_covariance_matrix = np.random.normal(size=(n_features, n_features))
    feature_covariance_matrix = feature_covariance_matrix @ feature_covariance_matrix.T

    genotype_dosage_variance = np.random.normal(size=(n_variants, 1))
    if one_dim:
        genotype_dosage_variance = genotype_dosage_variance.flatten()

    return (
        # Projection coefficients
        np.random.normal(size=(n_features, n_projections)),
        feature_covariance_matrix,
        genotype_dosage_variance,
        # Feature GWAS coefficients
        np.random.normal(size=(n_variants, n_features)),
        # Sample size
        np.random.randint(low=100, high=1e6),
    )


def build_data_pandas(n_features, n_projections, n_variants, one_dim, seed):
    T, P, s, B, N = build_data_numpy(**locals())
    feature_index = pd.Index(_create_names(n_features, "F"), name="feature")
    projection_index = pd.Index(_create_names(n_projections, "P"), name="projection")
    variant_index = pd.Index(_create_names(n_variants, "V"), name="variant")

    if one_dim:
        dosage = pd.Series(s, index=variant_index, name="s")
    else:
        dosage = pd.DataFrame(s, index=variant_index, columns=["s"])

    return (
        # Projection coefficients
        pd.DataFrame(T, index=feature_index, columns=projection_index),
        # Feature covariance
        pd.DataFrame(P, index=feature_index, columns=feature_index.rename("feature2")),
        # Genotype dosage variance
        dosage,
        # Feature GWAS coefficients
        pd.DataFrame(B, index=variant_index, columns=feature_index),
        # Sample size
        pd.DataFrame(N, index=variant_index, columns=projection_index),
    )


def validate_xarray_dataset(dataset):
    assert dataset.dims["feature"] == dataset.dims["feature2"]
    assert np.array_equal(dataset["feature"].values, dataset["feature2"].values)
    assert dataset["T"].dims[:2] == ("feature", "projection")
    assert dataset["P"].dims[:2] == ("feature", "feature2")
    assert dataset["s"].dims[0] == "variant"
    assert dataset["B"].dims[:2] == ("variant", "feature")


@pytest.mark.parametrize("n_features", [1, 5])
@pytest.mark.parametrize("n_projections", [1, 5])
@pytest.mark.parametrize("n_variants", [1, 5])
@pytest.mark.parametrize("one_dim", [True, False])
@pytest.mark.parametrize("seed", [0, 1])
def test_numpy_inputs(n_features, n_projections, n_variants, one_dim, seed):
    """Test that numpy inputs work, in general"""
    T, P, s, B, N = build_data_numpy(**locals())
    ds = build_xarray(T, P, s, B, N)
    validate_xarray_dataset(ds)
    features = [f"F{i}" for i in range(1, n_features + 1)]
    projections = [f"P{i}" for i in range(1, n_projections + 1)]
    variants = [f"V{i}" for i in range(1, n_variants + 1)]
    assert np.array_equal(features, ds["feature"].values)
    assert np.array_equal(projections, ds["projection"].values)

    assert np.array_equal(features, ds["feature"].values)
    assert np.array_equal(features, ds["feature2"].values)

    assert np.array_equal(variants, ds["variant"].values)

    assert np.array_equal(variants, ds["variant"].values)
    assert np.array_equal(features, ds["feature"].values)

    assert np.array_equal(variants, ds["variant"].values)
    assert np.array_equal(projections, ds["projection"].values)


@pytest.mark.parametrize("n_features", [1, 5])
@pytest.mark.parametrize("n_projections", [1, 5])
@pytest.mark.parametrize("n_variants", [1, 5])
@pytest.mark.parametrize("one_dim", [True, False])
@pytest.mark.parametrize("seed", [0, 1])
def test_pandas_inputs(n_features, n_projections, n_variants, one_dim, seed):
    """Test that pandas inputs work, in general"""
    T, P, s, B, N = build_data_pandas(**locals())
    ds = build_xarray(T, P, s, B, N)
    validate_xarray_dataset(ds)
    assert np.array_equal(T.index.values, ds["feature"].values)
    assert np.array_equal(T.columns.values, ds["projection"].values)

    assert np.array_equal(P.index.values, ds["feature"].values)
    assert np.array_equal(P.columns.values, ds["feature2"].values)

    assert np.array_equal(s.index.values, ds["variant"].values)

    assert np.array_equal(B.index.values, ds["variant"].values)
    assert np.array_equal(B.columns.values, ds["feature"].values)

    assert np.array_equal(N.index.values, ds["variant"].values)
    assert np.array_equal(N.columns.values, ds["projection"].values)


@pytest.mark.parametrize("n_features", [1, 5])
@pytest.mark.parametrize("n_projections", [1, 5])
@pytest.mark.parametrize("n_variants", [1, 5])
@pytest.mark.parametrize("one_dim", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_mixed_inputs(n_features, n_projections, n_variants, one_dim, seed):
    """Test that mixed numpy/pandas inputs work, in general"""

    # Copy the inputs
    args = dict(**locals())
    numpy_arrays = build_data_numpy(**args)
    pandas_arrays = build_data_pandas(**args)

    names = ["T", "P", "s", "B", "N"]
    kwargs = dict()

    np.random.seed(seed)
    for name, numpy, pandas in zip(names, numpy_arrays, pandas_arrays):
        if np.random.rand() < 0.5:
            kwargs[name] = numpy
        else:
            kwargs[name] = pandas

    ds = build_xarray(*kwargs.values())
    validate_xarray_dataset(ds)
