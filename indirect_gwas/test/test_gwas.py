import re

import pytest
import pandas as pd
import xarray as xr

from indirect_gwas.gwas import gwas_indirect
from .utils import ols_regression, build_dataset


def build_comparison_df(raw_dataset: xr.Dataset):
    (
        raw_dataset["BETA_direct"],
        raw_dataset["SE_direct"],
        raw_dataset["T_STAT_direct"],
        raw_dataset["P_direct"],
    ) = xr.apply_ufunc(
        ols_regression,
        raw_dataset["feature_phenotypes"] @ raw_dataset["T"],
        raw_dataset["G"],
        input_core_dims=[["sample"], ["sample"]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
    )

    (
        raw_dataset["BETA_indirect"],
        raw_dataset["SE_indirect"],
        raw_dataset["T_STAT_indirect"],
        raw_dataset["P_indirect"],
    ) = gwas_indirect(raw_dataset)

    comparison_df = raw_dataset[
        [
            "BETA_direct",
            "BETA_indirect",
            "SE_direct",
            "SE_indirect",
            "T_STAT_direct",
            "T_STAT_indirect",
            "P_direct",
            "P_indirect",
        ]
    ].to_dataframe()
    multiindex = pd.MultiIndex.from_tuples(
        comparison_df.columns.map(
            lambda x: re.split("_(?=direct|indirect)", x)
        ).values.tolist()
    )
    comparison_df = comparison_df.set_axis(multiindex, axis=1)
    return comparison_df


@pytest.mark.parametrize("seed", list(range(5)))
@pytest.mark.parametrize("maf", [0.01, 0.2, 0.5])
@pytest.mark.parametrize("n_samples", [1000])
@pytest.mark.parametrize("n_features", [1, 10])
@pytest.mark.parametrize("n_projections", [1, 10])
@pytest.mark.parametrize("n_variants", [1, 10])
def test_indirect_association_test(
    seed, maf, n_samples, n_features, n_projections, n_variants
):
    dataset = build_dataset(
        seed=seed,
        maf=maf,
        n_samples=n_samples,
        n_features=n_features,
        n_projections=n_projections,
        n_variants=n_variants,
    )
    ds = xr.Dataset(dataset)
    comparison_df = build_comparison_df(ds)
    for statistic in ["BETA", "SE", "T_STAT", "P"]:
        stat_df = comparison_df[statistic]
        diff = (stat_df["direct"] - stat_df["indirect"]).abs().max()
        assert diff == pytest.approx(0, abs=1e-5)
