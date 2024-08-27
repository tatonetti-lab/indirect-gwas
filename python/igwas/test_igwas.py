import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from igwas.igwas import igwas, igwas_files


@pytest.fixture
def test_data():
    return {
        "n_samples": 100,
        "n_variants": 100,
        "n_covariates": 10,
        "n_phenotypes": 10,
        "n_projections": 10,
    }


@pytest.fixture(scope="module")
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def setup_test_data(test_data, temp_dir):
    def _setup(compress=False) -> list[str]:
        # Generate test data
        np.random.seed(42)
        genotypes = np.random.randint(
            0, 3, size=(test_data["n_samples"], test_data["n_variants"])
        ).astype(np.float32)
        phenotypes = np.random.randn(
            test_data["n_samples"], test_data["n_phenotypes"]
        ).astype(np.float32)
        covariates = np.random.randn(
            test_data["n_samples"], test_data["n_covariates"]
        ).astype(np.float32)
        projection_matrix = np.random.normal(
            size=(test_data["n_phenotypes"], test_data["n_projections"])
        ).astype(np.float32)

        # Compute covariance matrix
        phenotype_residuals = (
            phenotypes
            - covariates @ np.linalg.lstsq(covariates, phenotypes, rcond=None)[0]
        )
        covariance_matrix = np.cov(phenotype_residuals.T)

        # Write files
        phenotype_names = [f"feat_{i}" for i in range(test_data["n_phenotypes"])]
        projection_names = [f"proj_{i}" for i in range(test_data["n_projections"])]
        phenotype_idx = pd.Index(phenotype_names, name="phenotype_id")
        projection_idx = pd.Index(projection_names, name="projection_id")
        pd.DataFrame(
            projection_matrix, columns=projection_idx, index=phenotype_idx
        ).to_csv(
            os.path.join(temp_dir, "projection_matrix.csv"),
        )
        pd.DataFrame(
            covariance_matrix, index=phenotype_idx, columns=phenotype_idx
        ).to_csv(
            os.path.join(temp_dir, "covariance_matrix.csv"),
        )

        # Generate GWAS results
        gwas_results = []
        for i in range(test_data["n_phenotypes"]):
            phenotype = phenotypes[:, i]
            results = []
            for j in range(test_data["n_variants"]):
                genotype = genotypes[:, j]
                beta, std_error = perform_direct_gwas(genotype, phenotype, covariates)
                results.append(
                    {
                        "variant_id": f"variant_{j}",
                        "beta": beta,
                        "std_error": std_error,
                        "sample_size": test_data["n_samples"],
                    }
                )

            file_name = f"feat_{i}.tsv" + (".zst" if compress else "")
            file_path = os.path.join(temp_dir, file_name)
            pd.DataFrame(results).to_csv(
                file_path,
                index=False,
                compression="zstd" if compress else None,
                sep="\t",
            )
            gwas_results.append(file_path)

        return gwas_results

    return _setup


def perform_direct_gwas(genotype, phenotype, covariates):
    X = np.column_stack((np.ones(len(genotype)), covariates, genotype))
    beta = np.linalg.lstsq(X, phenotype, rcond=None)[0]
    residuals = phenotype - X @ beta
    sigma2 = np.sum(residuals**2) / (len(phenotype) - X.shape[1])
    var_beta = sigma2 * np.linalg.inv(X.T @ X)[-1, -1]
    std_error = np.sqrt(var_beta)
    return beta[-1], std_error


@pytest.mark.parametrize("compress", [False, True])
def test_igwas_files(setup_test_data, test_data, temp_dir, compress):
    gwas_results = setup_test_data(compress)

    suffix = ".csv.zst" if compress else ".csv"
    output_file = os.path.join(temp_dir, f"igwas_results{suffix}")

    assert os.path.exists(os.path.join(temp_dir, "projection_matrix.csv"))
    assert os.path.exists(os.path.join(temp_dir, "covariance_matrix.csv"))

    for result in gwas_results:
        assert os.path.exists(result)

    igwas_files(
        projection_matrix_path=os.path.join(temp_dir, "projection_matrix.csv"),
        covariance_matrix_path=os.path.join(temp_dir, "covariance_matrix.csv"),
        gwas_result_paths=gwas_results,
        output_file_path=output_file,
        num_covar=test_data["n_covariates"],
        chunksize=test_data["n_variants"],
        variant_id="variant_id",
        beta="beta",
        std_error="std_error",
        sample_size="sample_size",
        num_threads=2,
        capacity=10,
        compress=compress,
        quiet=True,
    )

    results = pd.read_csv(output_file, sep="\t")
    assert results.shape[0] == test_data["n_variants"] * test_data["n_projections"]
    assert set(results.columns) == {
        "variant_id",
        "beta",
        "std_error",
        "t_stat",
        "neg_log_p_value",
        "sample_size",
    }


@pytest.mark.parametrize("compress", [False, True])
def test_igwas(setup_test_data, test_data, temp_dir, compress):
    gwas_results = setup_test_data(compress)

    suffix = ".csv.zst" if compress else ".csv"
    output_file = os.path.join(temp_dir, f"igwas_results{suffix}")

    assert os.path.exists(os.path.join(temp_dir, "projection_matrix.csv"))
    assert os.path.exists(os.path.join(temp_dir, "covariance_matrix.csv"))

    for result in gwas_results:
        assert os.path.exists(result)

    projection_matrix = pd.read_csv(
        os.path.join(temp_dir, "projection_matrix.csv"), index_col=0
    )
    covariance_matrix = pd.read_csv(
        os.path.join(temp_dir, "covariance_matrix.csv"), index_col=0
    )
    igwas(
        projection_matrix=projection_matrix,
        covariance_matrix=covariance_matrix,
        gwas_result_paths=gwas_results,
        output_file_path=output_file,
        num_covar=test_data["n_covariates"],
        chunksize=test_data["n_variants"],
        variant_id="variant_id",
        beta="beta",
        std_error="std_error",
        sample_size="sample_size",
        num_threads=2,
        capacity=10,
        compress=compress,
        quiet=True,
    )
