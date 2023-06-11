import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats
import xarray as xr


def ols_regression(y, X):
    reg = sm.OLS(y, sm.add_constant(X)).fit()
    B = reg.params[[1]]
    SE = reg.bse[[1]]
    T = B / SE
    P = 2 * scipy.stats.t.sf(np.abs(T), df=X.shape[0] - 1)
    return B.item(), SE.item(), T.item(), P.item()


def build_dataset(seed, maf, n_samples, n_features, n_projections, n_variants):
    np.random.seed(seed)

    sample_ids = pd.Index([f"S{i}" for i in range(1, n_samples + 1)], name="sample")
    variant_ids = pd.Index([f"V{i}" for i in range(1, n_variants + 1)], name="variant")
    feature_ids = pd.Index([f"F{i}" for i in range(1, n_features + 1)], name="feature")
    projection_ids = pd.Index(
        [f"P{i}" for i in range(1, n_projections + 1)], name="projection"
    )

    G = pd.DataFrame(
        np.random.binomial(2, maf, size=(n_samples, n_variants)),
        index=sample_ids,
        columns=variant_ids,
    )

    B_hidden = pd.DataFrame(
        np.random.normal(size=(n_variants, n_features)),
        index=variant_ids,
        columns=feature_ids,
    )

    error = np.random.normal(scale=0.01, size=(n_samples, n_features))
    feature_phenotypes = G @ B_hidden + error

    T = pd.DataFrame(
        np.random.uniform(low=-5, high=5, size=(n_features, n_projections)),
        index=feature_ids,
        columns=projection_ids,
    )
    P = (
        (feature_phenotypes - feature_phenotypes.mean(axis=0))
        .cov()
        .rename_axis("feature2", axis=1)
    )

    s = G.var(axis=0)

    B = (
        xr.apply_ufunc(
            ols_regression,
            xr.DataArray(feature_phenotypes),
            xr.DataArray(G),
            input_core_dims=[["sample"], ["sample"]],
            output_core_dims=[[], [], [], []],
            vectorize=True,
        )[0]
        .transpose("variant", "feature")
        .to_pandas()
    )

    N = pd.DataFrame(
        (
            np.repeat([n_samples], (n_variants * n_projections)).reshape(
                (n_variants, n_projections)
            )
        ),
        index=variant_ids,
        columns=projection_ids,
    )

    return {
        "G": G,
        "feature_phenotypes": feature_phenotypes,
        "T": T,
        "P": P,
        "s": s,
        "B": B,
        "df": N - 2,
    }
