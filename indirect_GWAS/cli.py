import argparse
import pathlib
import textwrap

import pandas as pd

from . import from_final_data, from_summary_statistics, gwas_indirect
from .io import compute_phenotypic_partial_covariance


def main():
    parser = argparse.ArgumentParser(
        prog="indirect-GWAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Compute indirect GWAS summary statistics

            Five main inputs are required:
            ------------------------------
            1. Projection coefficients (-P/--projection-coefficients)
            2. Feature partial covariance matrix (-C/--feature-partial-covariance)
            3. Genotype partial variance
                a. --standard-error-column or
                b. -S/--genotype-partial-variance
            4. Feature GWAS coefficient estimates (-gwas/--gwas-summary-statistics)
            5. Indirect GWAS degrees of freedom
                a. -dof/--degrees-of-freedom or
                b. -dof-column/--degrees-of-freedom-column or
                c. (-N/--sample-size or -N-column/--sample-size-column) and
                   (-K/--n-covar or -K-column/--n-covar-column))

            For more information about these choices, see the documentation at
            https://github.com/zietzm/indirect_GWAS/wiki.

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

    # 1. Directly specified for the indirect GWAS (either a number or a file giving
    #   variant-specific degrees of freedom)
    # 2. Computed using the sample size and number of covariates (both can be given
    #   either as a single number or a column name in the GWAS summary statistic files).
    #   If given through feature GWAS, the degrees of freedom will be computed as the
    #   variant-wise minimum across features.
    # This leads to the following outline of groups and arguments:
    # Direct DOF
    #   -dof
    # OR
    # Indirect DOF
    #   -dof-col
    #   OR
    #   Computed DOF
    #       Sample size
    #           -N
    #           OR
    #           -N-col
    #       AND
    #       Number of covariates
    #           -K
    #           OR
    #           -K-col
    parser.add_argument(
        "-dof",
        "--degrees-of-freedom",
        type=int,
        help="Degrees of freedom for the feature GWAS",
    )

    parser.add_argument(
        "-dof-column",
        "--degrees-of-freedom-column",
        type=str,
        help="""Column in the feature GWAS summary statistic files giving the degrees of
            freedom for each variant""",
    )

    sample_size_group = parser.add_mutually_exclusive_group()
    sample_size_group.add_argument(
        "-N",
        "--equal-sample-size",
        type=int,
        help="Sample size for the feature GWAS",
    )
    sample_size_group.add_argument(
        "-N-column",
        "--sample-size-column",
        type=str,
        help="""Column in the feature GWAS summary statistic files giving the sample
            size for each variant""",
    )

    n_covar_group = parser.add_mutually_exclusive_group()
    n_covar_group.add_argument(
        "-K",
        "--equal-number-of-covariates",
        type=int,
        help="""Number of covariates for the feature GWAS""",
    )
    n_covar_group.add_argument(
        "-K-column",
        "--number-of-covariates-column",
        type=str,
        help="""Column in the feature GWAS summary statistic files giving the number of
            covariates for each variant""",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        required=True,
        help="""Path to the output file(s). For every projection, a file will be created
            with the name <output>_<projection><extension>. Existing files will be
            overwritten.""",
    )

    parser.add_argument(
        "--extension",
        type=str,
        default=".csv",
        help="Extension to use for output files. E.g. '.csv', '.csv.gz', '.tsv', etc.",
    )

    parser.add_argument(
        "--float-format",
        type=str,
        default="%.15f",
        help="""Format string for writing floating point numbers. Passed directly to
            pandas.DataFrame.to_csv.""",
    )

    parser.add_argument(
        "--no-index",
        action="store_false",
        dest="has_index",
        help="""Whether to treat the first row and column of files as the names for the
            respective dimensions. If this flag is set, the first column and row will be
            treated as data, and the features/projections will be named 'feature_0',
            'feature_1', 'projection_0', etc.""",
    )

    parser.add_argument(
        "--separator",
        default=",",
        help="""Separator used to delimit fields in the input files. By default, this is
            a comma. To use a python escape like '\\t', use the $' bash syntax like
            `$'\\t'`.""",
    )

    args = parser.parse_args()

    # Check that the degrees of freedom are valid (since we can't use nested groups)
    if (args.degrees_of_freedom is None) and (args.degrees_of_freedom_column is None):
        if ((args.equal_sample_size is None) and (args.sample_size_column is None)) or (
            (args.equal_number_of_covariates is None)
            and (args.number_of_covariates_column is None)
        ):
            raise ValueError(
                "Either -dof, -dof-file, -dof-column, or -N/-N-column and -K/-K-column "
                "must be provided"
            )

    # Run the indirect gwas
    indirect_gwas_ds = load_data(args)
    (
        indirect_gwas_ds["beta"],
        indirect_gwas_ds["se"],
        indirect_gwas_ds["t_stat"],
        indirect_gwas_ds["p"],
    ) = gwas_indirect(indirect_gwas_ds)

    # Save the results
    save_data(
        indirect_gwas_ds, args.output, args.separator, args.extension, args.float_format
    )


def load_gwas_summary_statistics(
    files: list[pathlib.Path],
    variant_id_column: str,
    other_columns: list[str],
    separator: str = ",",
) -> pd.DataFrame:
    cols = [variant_id_column] + other_columns
    dataframes = []

    for file in files:
        df = (
            pd.read_csv(file, compression="infer", usecols=cols, sep=separator)
            .assign(feature=file.stem)
            .set_index(variant_id_column)
        )
        dataframes.append(df)

    return pd.concat(dataframes)


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

    # Gather the columns names that we need to load from each feature GWAS file
    gwas_columns = [
        args.coefficient_column,
        args.standard_error_column,
        args.degrees_of_freedom_column,
        args.sample_size_column,
        args.number_of_covariates_column,
    ]
    feature_gwas_df = load_gwas_summary_statistics(
        files=args.gwas_summary_statistics,
        variant_id_column=args.variant_id_column,
        other_columns=[col for col in gwas_columns if col is not None],
        separator=args.separator,
    )

    feature_gwas_coefficients = feature_gwas_df.pivot_table(
        index=args.variant_id_column, columns="feature", values=args.coefficient_column
    )

    # Load the degrees of freedom
    if args.degrees_of_freedom is not None:
        feature_gwas_dof = args.degrees_of_freedom
        projection_dof = args.degrees_of_freedom
    elif args.degrees_of_freedom_column is not None:
        feature_gwas_dof = feature_gwas_df.pivot_table(
            index=args.variant_id_column,
            columns="feature",
            values=args.degrees_of_freedom_column,
        )
        projection_dof = feature_gwas_dof.min(axis=1)
    else:
        if args.equal_sample_size is not None:
            sample_sizes = args.equal_sample_size
        else:
            sample_sizes = feature_gwas_df.pivot_table(
                index=args.variant_id_column,
                columns="feature",
                values=args.sample_size_column,
            )

        if args.equal_number_of_covariates is not None:
            number_of_covariates = args.equal_number_of_covariates
        else:
            number_of_covariates = feature_gwas_df.pivot_table(
                index=args.variant_id_column,
                columns="feature",
                values=args.number_of_covariates_column,
            )
        feature_gwas_dof = projection_dof = sample_sizes - number_of_covariates - 1

    # Compute the genotypic partial variance if needed
    if args.genotype_partial_variance is not None:
        genotype_partial_variance = pd.read_csv(
            args.genotype_partial_variance,
            header=header,
            index_col=index_col,
            sep=args.separator,
        )
        return from_final_data(
            projection_coefficients=projection_coefficients,
            feature_partial_covariance=feature_partial_covariance,
            feature_gwas_coefficients=feature_gwas_coefficients,
            genotype_partial_variance=genotype_partial_variance,
            projection_degrees_of_freedom=projection_dof,
        )
    else:
        feature_gwas_standard_error = feature_gwas_df.pivot_table(
            index=args.variant_id_column,
            columns="feature",
            values=args.standard_error_column,
        )
        return from_summary_statistics(
            projection_coefficients=projection_coefficients,
            feature_partial_covariance=feature_partial_covariance,
            feature_gwas_coefficients=feature_gwas_coefficients,
            feature_gwas_standard_error=feature_gwas_standard_error,
            feature_gwas_dof=feature_gwas_dof,
        )


def save_data(dataset, output_path, separator, output_extension, float_format):
    for projection in dataset["projection"].values:
        (
            dataset.sel(projection=projection)
            .drop("projection")[["beta", "se", "t_stat", "p"]]
            .to_dataframe()
            .to_csv(
                output_path.with_name(
                    f"{output_path.name}_{projection}{output_extension}"
                ),
                sep=separator,
                float_format=float_format,
            )
        )


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
