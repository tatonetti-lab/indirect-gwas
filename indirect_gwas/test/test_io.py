import numpy as np
import pandas as pd
import pytest

from indirect_gwas.io import from_final_data


def _create_names(n, prefix):
    return [f"{prefix}{i}" for i in range(1, n + 1)]


def build_data_numpy(n_features, n_projections, n_variants, seed):
    np.random.seed(seed)

    feature_covariance_matrix = np.random.normal(size=(n_features, n_features))
    feature_covariance_matrix = feature_covariance_matrix @ feature_covariance_matrix.T

    genotype_dosage_variance = np.random.normal(size=n_variants)

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


def build_data_pandas(n_features, n_projections, n_variants, seed):
    T, P, s, B, N = build_data_numpy(**locals())
    feature_index = pd.Index(_create_names(n_features, "F"), name="feature")
    projection_index = pd.Index(_create_names(n_projections, "P"), name="projection")
    variant_index = pd.Index(_create_names(n_variants, "V"), name="variant")

    return (
        # Projection coefficients
        pd.DataFrame(T, index=feature_index, columns=projection_index),
        # Feature covariance
        pd.DataFrame(P, index=feature_index, columns=feature_index.rename("feature2")),
        # Genotype dosage variance
        pd.Series(s, index=variant_index, name="s"),
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
@pytest.mark.parametrize("seed", [0, 1])
def test_pandas_inputs(n_features, n_projections, n_variants, seed):
    """Test that pandas inputs work, in general"""
    T, P, s, B, N = build_data_pandas(**locals())
    ds = from_final_data(
        projection_coefficients=T,
        feature_partial_covariance=P,
        feature_gwas_coefficients=B,
        genotype_partial_variance=s,
        projection_degrees_of_freedom=N.min(axis=1) - 2,
    )
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
