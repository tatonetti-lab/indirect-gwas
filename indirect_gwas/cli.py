import argparse
import datetime
import pathlib
import textwrap

import indirect_gwas._igwas


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
            4. Number of covariates (--number-of-covariates)

            For more information about these choices, see the documentation at
            https://github.com/tatonetti-lab/indirect-gwas/wiki.
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
        nargs="*",
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
        "--number-of-covariates",
        type=int,
        required=True,
        dest="n_covar",
        help="""Number of covariates variables for the feature GWAS. E.g.
            phenotype ~ 1 + genotype + age + sex + PC1 + PC2 would have 4 covariates""",
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
    #     help="""Format string for writing floating point numbers.""",
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
        help="""Number of rows to read at a time. This can be used to reduce memory
                usage.""",
    )

    parser.add_argument(
        "--single-file-output",
        action="store_true",
        help="""Whether to write all projections to a single file. If this flag is not
            set, a separate file will be created for each projection.""",
    )

    # parser.add_argument(
    #     "--computation-dtype",
    #     type=str,
    #     default="double",
    #     help="""Data type to use for computations.
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
        print(f"Started at\t{datetime.datetime.now().strftime('%c')}")

    # IDEA: Automatically sort files their order in the projection coefficients file
    # and/or feature partial covariance file. This would make wildcard expansion
    # feasible.

    paths = [p.as_posix() for p in args.gwas_summary_statistics]

    indirect_gwas._igwas.run(
        paths,
        args.variant_id_column,
        args.coefficient_column,
        args.standard_error_column,
        args.sample_size_column,
        args.projection_coefficients.as_posix(),
        args.feature_partial_covariance.as_posix(),
        args.output.as_posix(),
        args.n_covar,
        args.chunksize,
        args.single_file_output,
    )

    if not args.quiet:
        print(f"Finished at\t{datetime.datetime.now().strftime('%c')}")
