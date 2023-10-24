import pathlib
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
        results.append({
            "variant_id": variant,
            "beta": reg.params.iloc[0],
            "std_error": reg.bse.iloc[0],
            "t_statistic": reg.tvalues.iloc[0],
            "neg_log10_p_value": -np.log10(reg.pvalues.iloc[0]),
            "sample_size": len(X.index),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)


def test_cpp():
    # Get the mtcars R dataset
    with tempfile.TemporaryDirectory() as tmpdirname:
        raw_df = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data

        projection_names = ["P1", "P2"]

        # Feature phenotypes
        Y = raw_df[["mpg", "hp"]]
        feature_names = list(Y.columns)

        # Variants
        X = raw_df[["wt", "qsec"]]

        # Covariates
        covariates = raw_df[["cyl"]].assign(intercept=1)

        # Compute the feature GWAS
        feature_result_paths = list()
        for feature in feature_names:
            path = f"{tmpdirname}/{feature}.csv"
            gwas(Y[feature], X, covariates, path)
            feature_result_paths.append(path)

        # Projection coefficients
        beta = np.array([[1., 1.],
                         [1.5, 0.25]])

        beta_df = pd.DataFrame(beta, index=feature_names, columns=projection_names)
        beta_df.index.name = "feature_id"
        beta_df.to_csv(f"{tmpdirname}/projection_coefficients.csv")

        # Compute the feature partial covariance
        _feature_beta, _, _, _ = np.linalg.lstsq(covariates, Y, rcond=None)
        _feature_beta = pd.DataFrame(_feature_beta, index=covariates.columns,
                                     columns=feature_names)
        feature_partial_covariance = (Y - covariates @ _feature_beta).cov()
        try:
            feature_partial_covariance.index = feature_names
        except ValueError as e:
            print(feature_names, feature_partial_covariance.shape)
            raise e

        feature_partial_covariance.columns = feature_names
        feature_partial_covariance.index.name = "feature_id"
        feature_partial_covariance.to_csv(f"{tmpdirname}/feature_partial_covariance.csv")

        # Compute the indirect GWAS
        indirect_gwas._igwas.run(
            feature_result_paths,
            "variant_id",
            "beta",
            "std_error",
            "sample_size",
            f"{tmpdirname}/projection_coefficients.csv",
            f"{tmpdirname}/feature_partial_covariance.csv",
            f"{tmpdirname}/indirect",
            1,
            10,
        )

        # Compute the direct GWAS
        direct_result_paths = list()
        projection_df = pd.concat([Y @ beta_df, covariates], axis=1)
        for projection in projection_names:
            path = f"{tmpdirname}/direct_{projection}.csv"
            gwas(projection_df[projection], X, covariates, path)
            direct_result_paths.append(path)

        paths = list(pathlib.Path(tmpdirname).glob("*"))
        print("N paths produced: ", len(paths))

        for projection in projection_names:
            direct_df = pd.read_csv(f"{tmpdirname}/direct_{projection}.csv", index_col=0)
            indirect_df = pd.read_csv(f"{tmpdirname}/indirect_{projection}.csv", index_col=0)

            # Use pytest to check that all the columns are approximately equal
            for col in direct_df.columns:
                assert indirect_df[col].values == pytest.approx(direct_df[col].values, rel=1e-4)

    print("SUCCESS")


@pytest.mark.parametrize("t,df", [(0, 1), (1, 30), (1.5, 50), (-0.0696064, 29),
                                  (-1.02737, 29), (-3.11129, 29), (-3.66463, 29)])
def test_pvalues(t, df):
    print(f"TESTING {t}, {df}")
    python_version = (
        -1
        * (scipy.stats.t.logsf(np.abs(t), df) + np.log(2))
        / np.log(10)
    )
    cpp_version = indirect_gwas._igwas.compute_pvalue(t, df)
    print(python_version, cpp_version)
    assert cpp_version == pytest.approx(python_version)
