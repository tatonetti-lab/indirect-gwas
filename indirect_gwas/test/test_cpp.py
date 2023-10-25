import pathlib
import shlex
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest
import scipy.stats
import statsmodels.api as sm

import indirect_gwas


def gwas(y, X, covariates, output_path):
    """
    Run regressions on each feature in X against y, controlling for covariates.
    Save results to output_path.
    """
    exog_df = pd.concat([X, covariates], axis=1)

    results = list()
    for variant in X.columns:
        reg = sm.OLS(y, exog_df[[variant] + list(covariates.columns)]).fit()
        results.append(
            {
                "variant_id": variant,
                "beta": reg.params.iloc[0],
                "std_error": reg.bse.iloc[0],
                "t_statistic": reg.tvalues.iloc[0],
                "neg_log10_p_value": -np.log10(reg.pvalues.iloc[0]),
                "sample_size": len(X.index),
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, float_format="%.6f")


def compute_feature_partial_covariance(Y, covariates):
    """
    Compute the partial covariance of Y with respect to the covariates.
    Return a DataFrame with the feature names as the index and columns.
    """
    feature_names = list(Y.columns)

    _feature_beta, _, _, _ = np.linalg.lstsq(covariates, Y, rcond=None)
    _feature_beta = pd.DataFrame(
        _feature_beta, index=covariates.columns, columns=feature_names
    )
    feature_partial_covariance = (Y - covariates @ _feature_beta).cov()
    try:
        feature_partial_covariance.index = feature_names
    except ValueError as e:
        print(feature_names, feature_partial_covariance.shape)
        raise e

    feature_partial_covariance.columns = feature_names
    feature_partial_covariance.index.name = "feature_id"
    return feature_partial_covariance


def setup_test_data(temporary_directory):
    """
    Setup a test dataset with 2 features and 2 variants. Save all necessary
    files to temporary_directory.
    """
    raw_df = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data

    feature_names = ["mpg", "hp"]
    variant_names = ["wt", "qsec"]
    projection_names = ["P1", "P2"]

    # Feature phenotypes
    Y = raw_df[feature_names]

    # Variants
    X = raw_df[variant_names]

    # Covariates
    covariates = raw_df[["cyl"]].assign(intercept=1)

    # Compute the feature GWAS
    feature_result_paths = list()
    for feature in feature_names:
        path = f"{temporary_directory}/{feature}.csv"
        gwas(Y[feature], X, covariates, path)
        feature_result_paths.append(path)

    # Projection coefficients
    beta = np.array([[1.0, 1.0], [1.5, 0.25]])
    beta_df = pd.DataFrame(beta, index=feature_names, columns=projection_names)
    beta_df.index.name = "feature_id"
    beta_df.to_csv(f"{temporary_directory}/projection_coefficients.csv")

    # Compute the feature partial covariance
    feature_partial_covariance = compute_feature_partial_covariance(Y, covariates)
    feature_partial_covariance.to_csv(
        f"{temporary_directory}/feature_partial_covariance.csv",
        float_format="%.6f",
    )

    return dict(
        feature_result_paths=feature_result_paths,
        projection_names=projection_names,
        beta_df=beta_df,
        feature_partial_covariance=feature_partial_covariance,
        Y=Y,
        X=X,
        covariates=covariates,
    )


def compare_direct_vs_indirect(tmpdirname, data):
    """
    Compare the direct and indirect GWAS results.
    Ensure that results are approximately equal (up to 1e-5)
    """
    # Compute the direct GWAS
    direct_result_paths = list()
    projection_df = pd.concat([data["Y"] @ data["beta_df"], data["covariates"]], axis=1)
    for projection in data["projection_names"]:
        path = f"{tmpdirname}/direct_{projection}.csv"
        gwas(projection_df[projection], data["X"], data["covariates"], path)
        direct_result_paths.append(path)

    for projection in data["projection_names"]:
        direct_df = pd.read_csv(f"{tmpdirname}/direct_{projection}.csv", index_col=0)
        indirect_df = pd.read_csv(
            f"{tmpdirname}/indirect_{projection}.csv", index_col=0
        )

        # Use pytest to check that all the columns are approximately equal
        for col in direct_df.columns:
            max_diff = np.abs(indirect_df[col].values - direct_df[col].values).max()
            assert max_diff == pytest.approx(0, abs=1e-4, rel=1e-4)

    paths = list(pathlib.Path(tmpdirname).glob("indirect*"))
    return paths


@pytest.mark.parametrize("chunksize", [1, 2, 10, 25, 1000])
def test_cpp(chunksize):
    """
    Test the C++ code using the Python API.
    Ensure that results are approximately equal to the direct GWAS (up to 1e-5)
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        data = setup_test_data(tmpdirname)

        indirect_gwas._igwas.run(
            data["feature_result_paths"],
            "variant_id",
            "beta",
            "std_error",
            "sample_size",
            f"{tmpdirname}/projection_coefficients.csv",
            f"{tmpdirname}/feature_partial_covariance.csv",
            f"{tmpdirname}/indirect",
            1,
            chunksize,
        )

        paths = compare_direct_vs_indirect(tmpdirname, data)

    print(f"SUCCESS! Direct use produced {len(paths)} paths")


@pytest.mark.parametrize("chunksize", [1, 2, 10, 25, 1000])
def test_cpp_cli(chunksize):
    """
    Test the C++ code using the CLI.
    Ensure that results are approximately equal to the direct GWAS (up to 1e-5)
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        data = setup_test_data(tmpdirname)

        result = subprocess.run(
            shlex.split(
                f"""
                indirect-gwas
                -P {tmpdirname}/projection_coefficients.csv
                -C {tmpdirname}/feature_partial_covariance.csv
                --variant-id-column variant_id
                --coefficient-column beta
                --standard-error-column std_error
                --sample-size-column sample_size
                --number-of-exogenous 1
                --chunksize 10
                --gwas-summary-statistics {" ".join(data["feature_result_paths"])}
                --output {tmpdirname}/indirect
                """
            )
        )

        if not result.returncode == 0:
            print(result.stdout)
            print(result.stderr)

        paths = compare_direct_vs_indirect(tmpdirname, data)

    print(f"SUCCESS! CLI use produced {len(paths)} paths")


@pytest.mark.parametrize(
    "t,df",
    [
        (0, 1),
        (1, 30),
        (1.5, 50),
        (-0.0696064, 29),
        (-1.02737, 29),
        (-3.11129, 29),
        (-3.66463, 29),
    ],
)
def test_pvalues(t, df):
    print(f"TESTING {t}, {df}")
    python_version = -1 * (scipy.stats.t.logsf(np.abs(t), df) + np.log(2)) / np.log(10)
    cpp_version = indirect_gwas._igwas.compute_pvalue(t, df)
    print(python_version, cpp_version)
    assert cpp_version == pytest.approx(python_version)
