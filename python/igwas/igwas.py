import os
import tempfile

import pandas as pd

from igwas._lowlevel import igwas_impl


def igwas_files(
    projection_matrix_path: str,
    covariance_matrix_path: str,
    gwas_result_paths: list[str],
    output_file_path: str,
    num_covar: int = 1,
    chunksize: int = 100_000,
    variant_id: str = "ID",
    beta: str = "BETA",
    std_error: str = "SE",
    sample_size: str = "OBS_CT",
    num_threads: int = 1,
    capacity: int = 25,
    compress: bool = False,
    quiet: bool = False,
    write_phenotype_id: bool = False,
) -> None:
    """Run Indirect GWAS

    Indirect GWAS is a method for computing summary statistics for a GWAS on
    a linear combination of phenotypes. For more information, see
    https://github.com/tatonetti-lab/indirect-gwas.

    Args:
        projection_matrix_path: Path to the projection matrix
        covariance_matrix_path: Path to the covariance matrix
        gwas_result_paths: List of paths to GWAS results
        output_file_path: Path to the output file
        num_covar: Number of covariates
        chunksize: Number of variants to process at a time
        variant_id: Name of the variant ID column
        beta: Name of the beta column
        std_error: Name of the standard error column
        sample_size: Name of the sample size column
        num_threads: Number of threads to use
        capacity: Capacity of the queue (max number of files to process at a time)
        compress: Whether to compress the output file
        quiet: Whether to suppress output

    Returns:
        None
    """
    igwas_impl(
        projection_matrix_path,
        covariance_matrix_path,
        gwas_result_paths,
        output_file_path,
        num_covar,
        chunksize,
        variant_id,
        beta,
        std_error,
        sample_size,
        num_threads,
        capacity,
        compress,
        quiet,
        write_phenotype_id,
    )


def igwas(
    projection_matrix: pd.DataFrame,
    covariance_matrix: pd.DataFrame,
    gwas_result_paths: list[str],
    output_file_path: str,
    num_covar: int = 1,
    chunksize: int = 100_000,
    variant_id: str = "ID",
    beta: str = "BETA",
    std_error: str = "SE",
    sample_size: str = "OBS_CT",
    num_threads: int = 1,
    capacity: int = 25,
    compress: bool = False,
    quiet: bool = False,
):
    """Run Indirect GWAS

    Indirect GWAS is a method for computing summary statistics for a GWAS on
    a linear combination of phenotypes. For more information, see
    https://github.com/tatonetti-lab/indirect-gwas.

    Args:
        projection_matrix: Projection matrix
        covariance_matrix: Covariance matrix
        gwas_result_paths: List of paths to GWAS results
        output_file_path: Path to the output file
        num_covar: Number of covariates
        chunksize: Number of variants to process at a time
        variant_id: Name of the variant ID column
        beta: Name of the beta column
        std_error: Name of the standard error column
        sample_size: Name of the sample size column
        num_threads: Number of threads to use
        capacity: Capacity of the queue (max number of files to process at a time)
        compress: Whether to compress the output file
        quiet: Whether to suppress output

    Returns:
        None
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        projection_matrix_path = os.path.join(tmp_dir, "projection_matrix.csv")
        covariance_matrix_path = os.path.join(tmp_dir, "covariance_matrix.csv")
        projection_matrix.to_csv(projection_matrix_path)
        covariance_matrix.to_csv(covariance_matrix_path)
        igwas_files(
            projection_matrix_path,
            covariance_matrix_path,
            gwas_result_paths=gwas_result_paths,
            output_file_path=output_file_path,
            num_covar=num_covar,
            chunksize=chunksize,
            variant_id=variant_id,
            beta=beta,
            std_error=std_error,
            sample_size=sample_size,
            num_threads=num_threads,
            capacity=capacity,
            compress=compress,
            quiet=quiet,
        )
