import argparse
import datetime
import os
import pathlib

import numpy as np
import pandas as pd
import polars as pl
import scipy.stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gwas-sumstats",
        type=pathlib.Path,
        nargs="+",
        required=True,
        help=(
            "List of GWAS summary statistic files. Wildcards like result.*.glm may be"
            " used"
        ),
    )
    parser.add_argument(
        "--projection-coefficients",
        type=pathlib.Path,
        required=True,
        help="Path to the projection coefficients file.",
    )
    parser.add_argument(
        "--feature-partial-covariance",
        type=pathlib.Path,
        required=True,
        help="Path to the feature partial covariance file.",
    )
    parser.add_argument(
        "--n-exogenous",
        type=int,
        required=True,
        help=(
            "Number of exogeneous variables in the projection model (i.e. number of"
            " covariates + 1)."
        ),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help=(
            "Path to the output file. If --no-split is not specified, the output"
            " will be split into separate files, one for each projection."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Chunk size for processing the GWAS files in number of variants",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Flag to indicate whether to split the output into separate files.",
    )
    parser.add_argument(
        "--variant-id-column",
        default="ID",
        help="Name of the column containing variant IDs in the input files.",
    )
    parser.add_argument(
        "--beta-column",
        default="BETA",
        help="Name of the column containing beta values in the input files.",
    )
    parser.add_argument(
        "--se-column",
        default="SE",
        help="Name of the column containing standard error values in the input files.",
    )
    parser.add_argument(
        "--n-column",
        default="OBS_CT",
        help=(
            "Name of the column containing observation count values in the input"
            " files."
        ),
    )
    parser.add_argument(
        "--computation-dtype",
        default="float32",
        help=(
            "Data type used for computations. Float32 will use less memory and"
            " disk, but results in decreased precision."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Flag to enable quiet output.",
    )
    return parser.parse_args()


class IndirectGWAS:
    def __init__(
        self,
        gwas_summary_statistics: list[os.PathLike],
        projection_coefficients: np.ndarray | pd.DataFrame,
        feature_partial_covariance: np.ndarray | pd.DataFrame,
        n_exogenous: int,
        output: os.PathLike,
        chunksize: int = 100_000,
        variant_id_column: str = "ID",
        beta_column: str = "BETA",
        se_column: str = "SE",
        n_column: str = "OBS_CT",
        computation_dtype: str = "float32",
        quiet: bool = False,
    ):
        # Indirect GWAS data
        self.gwas_summary_statistics = gwas_summary_statistics
        self.projection_coefficients = projection_coefficients
        self.covariance = feature_partial_covariance
        self.n_exogenous = n_exogenous

        # Options
        self.output = pathlib.Path(output)
        self.chunksize = chunksize
        self.computation_dtype = computation_dtype
        self.quiet = quiet

        # Column names
        self.variant_id_column = variant_id_column
        self.beta_column = beta_column
        self.se_column = se_column
        self.n_column = n_column
        self.zfill = (np.floor(np.log10(len(self.gwas_summary_statistics))) + 1).astype(
            int
        )

        # Check inputs and gather statistics
        self.projection_ids, self.variant_ids = self.check_inputs()

        # Derived quantities
        self.n_features = len(gwas_summary_statistics)
        self.n_projections = len(self.projection_ids)
        self.n_variants = len(self.variant_ids)
        self.feature_partial_variance = np.diag(feature_partial_covariance)
        self.var_p_p = np.diag(
            projection_coefficients.T
            @ feature_partial_covariance
            @ projection_coefficients
        ).reshape(-1, 1)
        self.memory_map_temporary_path = None
        self.mmap = None

        if not self.quiet:
            print(f"Found {self.n_features} features")
            print(f"Found {self.n_projections} projections")
            print(f"Found {self.n_variants} variants")

    def check_inputs(self):
        # Check that all gwas_sumstats files exist
        for path in self.gwas_summary_statistics:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

        # Check that all gwas_sumstat files have exactly the same variant IDs in the
        # same order
        variant_ids = None
        for path in self.gwas_summary_statistics:
            file_variant_ids = pl.read_csv(
                path, separator="\t", columns=[self.variant_id_column]
            )[self.variant_id_column].to_numpy()
            if variant_ids is None:
                variant_ids = file_variant_ids
            else:
                if not np.array_equal(variant_ids, file_variant_ids):
                    raise ValueError(
                        f"Variant IDs in {path} do not match those in the other files"
                    )

        # Check that the projection coefficients have the same number of rows as the
        # number of files in gwas_sumstats
        if isinstance(self.projection_coefficients, pd.DataFrame | np.ndarray):
            if self.projection_coefficients.shape[0] != len(
                self.gwas_summary_statistics
            ):
                raise ValueError(
                    "Number of rows in projection coefficients does not match the "
                    "number of files in gwas_sumstats"
                )
        else:
            raise TypeError("projection_coefficients must be an array or DataFrame")

        # Check that the feature partial covariance has the same number of rows as the
        # number of files in gwas_sumstats
        if isinstance(self.covariance, pd.DataFrame | np.ndarray):
            if self.covariance.shape[0] != len(self.gwas_summary_statistics):
                raise ValueError(
                    "Number of rows in feature partial covariance does not match the "
                    "number of files in gwas_sumstats"
                )
        else:
            raise TypeError("feature_partial_covariance must be an array or DataFrame")

        # Get the projection names
        if isinstance(self.projection_coefficients, pd.DataFrame):
            projection_ids = self.projection_coefficients.columns
        else:
            projection_ids = [
                f"projection_{str(i).zfill(self.zfill)}"
                for i in range(self.projection_coefficients.shape[1])
            ]

        return projection_ids, variant_ids

    def create_mmap_file(self):
        """
        Shape is (6, n_projections, n_variants)
        0: degrees-of-freedom
        1: beta
        2: sum of genotype partial variances
        3: standard error
        4: t-statistic
        5: p-value
        """
        self.memory_map_temporary_path = self.output.parent.joinpath(
            f"{self.output.stem}_temp.mmap"
        )

        self.mmap = np.memmap(
            self.memory_map_temporary_path,
            dtype=self.computation_dtype,
            mode="w+",
            shape=(6, self.n_projections, self.n_variants),
        )
        self.mmap[0] = np.inf
        self.mmap[1:] = 0.0

    def process_file_chunked(
        self,
        feature_gwas_path: os.PathLike,
        feature_projection_coefficients: np.ndarray,
        feature_partial_variance: float,
    ):
        reader = pl.read_csv_batched(
            feature_gwas_path,
            separator="\t",
            has_header=True,
            batch_size=self.chunksize,
            columns=["BETA", "SE", "OBS_CT"],
        )
        last_index = 0
        while True:
            chunks = reader.next_batches(1)
            if chunks is None:
                break

            chunk = chunks[0]

            beta = chunk["BETA"].to_numpy().astype(self.computation_dtype)
            se = chunk["SE"].to_numpy().astype(self.computation_dtype)
            N = chunk["OBS_CT"].to_numpy().astype(self.computation_dtype)

            current_chunk = self.mmap[:, :, last_index : last_index + len(chunk)]

            # Update the degrees-of-freedom
            dof = N - self.n_exogenous - 1
            current_chunk[0] = np.minimum(current_chunk[0], dof)

            # Update beta
            current_chunk[1] += np.outer(feature_projection_coefficients, beta)

            # Update the sum of genotype partial variances
            current_chunk[2] += feature_partial_variance / (dof * se**2 + beta**2)
            last_index += len(chunk)

        self.mmap.flush()

    def make_chunk_df(self, chunk: np.memmap, variant_ids: list[str]):
        output_df = pd.DataFrame(
            {
                "projection": np.tile(self.projection_ids, chunk.shape[2]),
                "variant": np.repeat(variant_ids, self.n_projections),
                "beta": chunk[1].ravel("F"),
                "se": chunk[3].ravel("F"),
                "t": chunk[4].ravel("F"),
                "p": chunk[5].ravel("F"),
                "dof": chunk[0].ravel("F"),
            }
        )
        return output_df

    def write_output_chunked(self):
        chunk_indices = list(range(0, self.n_variants, self.chunksize))
        for chunk_idx in chunk_indices:
            chunk = self.mmap[:, :, chunk_idx : chunk_idx + self.chunksize]
            variant_ids = self.variant_ids[chunk_idx : chunk_idx + self.chunksize]

            output_df = self.make_chunk_df(chunk, variant_ids)
            for name, group_df in output_df.groupby("projection"):
                output_path = self.output.with_name(
                    f"{self.output.stem}_{name}{self.output.suffix}"
                )
                group_df.to_csv(
                    output_path,
                    mode=("w" if chunk_idx == 0 else "a"),
                    header=(chunk_idx == 0),
                    sep="\t",
                    index=False,
                    chunksize=100_000,
                )

    def run(self):
        # Create the memory map temporary file
        if not self.quiet:
            print("Opening a temporary file to store computations")

        if self.mmap is None:
            self.create_mmap_file()

        # Process each GWAS summary statistic file, adding the results to the memory map
        for i, path in enumerate(self.gwas_summary_statistics):
            if not self.quiet:
                name = str(i + 1).zfill(self.zfill)
                print(
                    f"Processing file {name}/{len(self.gwas_summary_statistics)}: {path}"
                )

            self.process_file_chunked(
                path,
                np.array(self.projection_coefficients)[i],
                np.array(self.feature_partial_variance)[i],
            )

        # Compute summary statistics
        if not self.quiet:
            print("Computing final summary statistics")
        # Genotypic partial variance is taken as the mean across features
        self.mmap[2] = self.mmap[2] / self.n_features
        # Standard error
        self.mmap[3] = np.sqrt(
            (self.var_p_p / self.mmap[2] - self.mmap[1] ** 2) / self.mmap[0]
        )
        # t-statistic
        self.mmap[4] = self.mmap[1] / self.mmap[3]
        # p-value
        self.mmap[5] = -np.log10(
            2 * scipy.stats.t.sf(np.abs(self.mmap[4]), self.mmap[0])
        )
        self.mmap.flush()

        # Write the output
        if not self.quiet:
            print("Completed computations, generating output file(s)")

        self.write_output_chunked()

        # Clean up the memory map temporary file
        if not self.quiet:
            print("Finished writing output, cleaning up temporary file")

        os.remove(self.memory_map_temporary_path)


def main():
    args = parse_args()

    if not args.quiet:
        # Print start datetime in human-readable format
        print(f"Started at {datetime.datetime.now().strftime('%c')}")

    # Read the projection coefficients
    projection_coefficients = pd.read_csv(
        args.projection_coefficients, sep="\t", index_col=0
    )

    # Read the feature partial covariance
    feature_partial_covariance = pd.read_csv(
        args.feature_partial_covariance, sep="\t", index_col=0
    )

    # Run the indirect GWAS
    indirect_gwas = IndirectGWAS(
        args.gwas_sumstats,
        projection_coefficients,
        feature_partial_covariance,
        args.n_exogenous,
        args.output,
        chunksize=args.chunksize,
        no_split=args.no_split,
        variant_id_column=args.variant_id_column,
        beta_column=args.beta_column,
        se_column=args.se_column,
        n_column=args.n_column,
        computation_dtype=args.computation_dtype,
        memory_map_temporary_path=args.memory_map_temporary_path,
        quiet=args.quiet,
    )
    indirect_gwas.run()

    if not args.quiet:
        print(f"Finished at {datetime.datetime.now().strftime('%c')}")
