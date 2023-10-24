import argparse
import datetime
import pathlib
import textwrap

import pandas as pd

from .io import compute_phenotypic_partial_covariance
import _igwas


def parse_args():
    parser = argparse.ArgumentParser(
        prog="indirect-gwas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Compute indirect GWAS summary statistics

            Four main inputs are required:
            ------------------------------
            1. Projection coefficients (-P/--projection-coefficients)
            2. Feature partial covariance matrix (-C/--feature-partial-covariance)
            3. Feature GWAS summary statistics (-gwas/--gwas-summary-statistics)
                a. --variant-id-column
                b. --coefficient-column
                c. --standard-error-column
                d. --sample-size-column
            4. Number of exogenous variables (-n/--number-of-exogenous)

            For more information about these choices, see the documentation at
            https://github.com/tatonetti-lab/indirect-gwas/wiki.

            All paths are passed directly to pandas.read_csv with compression='infer'.
            By default, the first column is used as the index, and the first row is used
            as the column names. To disable this behavior, use the --no-index flag.
            """
        ),
    )

    parser.add_argument(
        "-P",
        "--projection-coefficients",
        type=pathlib.Path,
        required=True,
        help="Path to a file with the projection coefficients (features x projections).",
    )
    parser.add_argument(
        "-C",
        "--feature-partial-covariance",
        required=True,
        type=pathlib.Path,
        help="Path to feature partial covariance matrix file (features x features)",
    )

    # Feature GWAS summary statistics
    parser.add_argument(
        "-gwas",
        "--gwas-summary-statistics",
        type=pathlib.Path,
        required=True,
        nargs="+",
        help="""Path(s) to GWAS summary statistics for the feature phenotypes.
            Separate files with spaces. File name stems will be treated as the feature
            names. Can include wildcards like 'path/to/files/*.csv.gz'. The variant
            ID column must be specified with --variant-id-col.""",
    )
    parser.add_argument(
        "--variant-id-column",
        type=str,
        required=True,
        help="""Column in GWAS summary statistic files used to unique identify variants.
            For example, 'RSID', 'CHR:POS', 'CHR:POS:A1:A2', etc.""",
    )
    parser.add_argument(
        "--coefficient-column",
        type=str,
        required=True,
        help="""Column in GWAS summary statistic files giving coefficient estimates.
            For example, 'BETA', 'coef', etc.""",
    )
    parser.add_argument(
        "--standard-error-column",
        required=True,
        type=str,
        help="""Column in the feature GWAS summary statistic files giving the standard
            error of the coefficient.""",
    )
    parser.add_argument(
        "--sample-size-column",
        type=str,
        required=True,
        help="""Column in the feature GWAS summary statistic files giving the sample
            size for each variant""",
    )

    parser.add_argument(
        "--number-of-exogenous",
        type=int,
        required=True,
        dest="n_exogenous",
        help="""Number of exogenous variables for the feature GWAS. E.g.
            phenotype ~ 1 + genotype + age + sex + PC1 + PC2 would have 5 exogenous""",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        required=True,
        dest="output",
        help="""Path to the output file(s). For every projection, a file will be created
            with the name <output>_<projection><extension>. Existing files will be
            overwritten.""",
    )

    # parser.add_argument(
    #     "--float-format",
    #     type=str,
    #     default="%.15f",
    #     help="""Format string for writing floating point numbers. Passed directly to
    #         pandas.DataFrame.to_csv.""",
    # )

    # parser.add_argument(
    #     "--no-index",
    #     action="store_false",
    #     dest="has_index",
    #     help="""Whether to treat the first row and column of files as the names for the
    #         respective dimensions. If this flag is set, the first column and row will be
    #         treated as data, and the features/projections will be named 'feature_0',
    #         'feature_1', 'projection_0', etc.""",
    # )

    # parser.add_argument(
    #     "--separator",
    #     default="\t",
    #     help="""Separator used to delimit fields in the input files. By default, this is
    #         a comma. To use a python escape like '\\t', use the $' bash syntax like
    #         `$'\\t'`.""",
    # )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="""Number of rows to read at a time. Passed directly to pandas.read_csv.
            This can be used to reduce memory usage.""",
    )

    # parser.add_argument(
    #     "--computation-dtype",
    #     type=str,
    #     default="double",
    #     help="""Data type to use for computations. Passed directly to pandas.read_csv.
    #         This can be used to reduce memory usage. E.g. 'float32', 'float64',
    #         'single', 'double', 'half', etc.""",
    # )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Whether to suppress progress messages.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not args.quiet:
        # Print start datetime in human-readable format
        print(f"Started at {datetime.datetime.now().strftime('%c')}")

    _igwas.run(
        args.gwas_summary_statistics,
        args.variant_id_column,
        args.coefficient_column,
        args.standard_error_column,
        args.sample_size_column,
        args.projection_coefficients,
        args.feature_partial_covariance,
        args.output,
        args.n_exogenous,
        args.chunksize,
    )

    if not args.quiet:
        print(f"Finished at {datetime.datetime.now().strftime('%c')}")


def load_data(args):
    if args.has_index:
        index_col = 0
        header = 0
    else:
        index_col = None
        header = None

    projection_coefficients = pd.read_csv(
        args.projection_coefficients,
        header=header,
        index_col=index_col,
        sep=args.separator,
    )

    feature_partial_covariance = pd.read_csv(
        args.feature_partial_covariance,
        header=header,
        index_col=index_col,
        sep=args.separator,
    )
    return projection_coefficients, feature_partial_covariance


def compute_feature_partial_covariance():
    parser = argparse.ArgumentParser(
        prog="compute-feature-partial-covariance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Compute the feature partial covariance matrix from individual-level data

            This is a helper script for when individual-level data are available, and
            GWAS are run using simple linear models.

            All paths are passed directly to pandas.read_csv with compression='infer'.
            By default, the first column is used as the index, and the first row is used
            as the column names. To disable this behavior, use the --no-index flag.
            """
        ),
    )

    parser.add_argument(
        "--pheno",
        type=pathlib.Path,
        required=True,
        help="Path to individual-level feature phenotypes file (samples x features)",
    )
    parser.add_argument(
        "--pheno-name",
        nargs="*",
        type=str,
        help="""Name(s) of the feature(s) in the phenotype file. Default is all columns
            except the sample ID columns (or, if --covar is not passed, default is all
            columns except sample ID columns and --covar-name columns).
            E.g. --pheno-name BMI""",
    )

    parser.add_argument(
        "--covar",
        type=pathlib.Path,
        help="""Path to individual-level covariates file (samples x covariates).
            If not provided, assumes covariates are in the phenotype file and identified
            by --covar-name.""",
    )
    parser.add_argument(
        "--covar-name",
        nargs="*",
        type=str,
        help="""Name(s) of the covariates(s) in the covariates file (or in the --pheno
            file if --covar is not provided). Default is all columns in --covar except
            the sample ID columns. E.g. --covar-name age sex PC1""",
    )

    parser.add_argument(
        "--sample-id-column",
        nargs="+",
        type=str,
        default=["#FID", "IID"],
        help="""Column(s) in the phenotype and covariate files that uniquely identify
            each sample. E.g. --sample-id-column $'#FID' IID""",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        required=True,
        help="Path to output file",
    )

    parser.add_argument(
        "--add-intercept",
        action="store_true",
        help="Whether to add an intercept to the covariates",
    )
    parser.add_argument(
        "--separator",
        default=",",
        help="""Separator used to delimit fields in the input files. By default, this is
            a comma. To use a python escape like '\\t', use the $' bash syntax like
            `$'\\t'`.""",
    )
    parser.add_argument(
        "--float-format",
        type=str,
        default="%.15f",
        help="""Format string for writing floating point numbers. Passed directly to
            pandas.DataFrame.to_csv.""",
    )
    args = parser.parse_args()

    # Load the minimum number of columns into memory
    if args.pheno_name is not None and args.covar is not None:
        # Load only the sample ID and pheno_name columns
        pheno_cols = args.sample_id_column + args.pheno_name
    elif args.pheno_name is not None and args.covar is None:
        # Load only the sample ID, pheno_name, and covar_name columns
        pheno_cols = args.sample_id_column + args.pheno_name + args.covar_name
    else:
        # Load all columns
        pheno_cols = None

    # Load the data
    feature_phenotypes = pd.read_csv(
        args.pheno, sep=args.separator, usecols=pheno_cols
    ).set_index(args.sample_id_column)

    if args.covar is not None:
        # Load the minimum number of columns into memory
        if args.cover_name is not None:
            # Load only the sample ID and covar_name columns
            covar_cols = args.sample_id_column + args.covar_name
        else:
            # Load all columns
            covar_cols = None

        covariates = (
            pd.read_csv(args.covar, sep=args.separator, usecols=covar_cols)
            .set_index(args.sample_id_column)
            .loc[feature_phenotypes.index]
        )
    else:
        # Covariates are found in the phenotype file
        covariates = feature_phenotypes[args.covar_name]
        if args.pheno_name is not None:
            # Feature phenotypes are defined by the user
            feature_phenotypes = feature_phenotypes[args.pheno_name]
        else:
            # Feature phenotypes are all columns except the sample ID and covariates
            feature_phenotypes = feature_phenotypes.drop(args.sample_id_column, axis=1)

    if args.add_intercept:
        covariates = covariates.assign(intercept=1.0)

    feature_partial_covariance = compute_phenotypic_partial_covariance(
        feature_phenotypes=feature_phenotypes,
        covariates=covariates,
        add_intercept=args.add_intercept,
    )

    feature_partial_covariance.to_csv(
        args.output, sep=args.separator, float_format=args.float_format
    )
