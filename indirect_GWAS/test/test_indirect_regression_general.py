import numpy as np
import xarray as xr
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from indirect_GWAS.gwas import gwas_indirect


def build_xarray(T, P, s, B, N):
    data = xr.Dataset(
        {
            "T": (["feature", "projection"], T),
            "P": (["feature", "feature2"], P),
            "s": (["variant"], s),
            "B": (["variant", "feature"], B),
            "N": (["variant", "projection"], N),
        }
    )
    return data


def compute_direct_regression(data, x_col, y_col, w):
    data['z'] = np.dot(data[y_col], w)
    result = smf.ols(formula=f'z ~ {x_col}', data=data).fit()
    return result.params, result.bse


@pytest.mark.parametrize("dataset_name", [
    "engel",
    "grunfeld"
])
@pytest.mark.parametrize("random_seed", list(range(5)))
def test_gwas_indirect(dataset_name, random_seed):
    # Load the dataset
    if dataset_name == "engel":
        dataset = sm.datasets.engel.load_pandas().data
    elif dataset_name == "grunfeld":
        dataset = sm.datasets.grunfeld.load_pandas().data
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    dataset = dataset.select_dtypes(include=["number"])

    # Select x and y columns
    columns = dataset.columns
    x_col = columns[0]
    y_cols = columns[1:]

    # Run the regressions of y ~ x
    B = []
    P = dataset[y_cols].cov().values
    for y_col in y_cols:
        model = smf.ols(formula=f'{y_col} ~ {x_col}', data=dataset)
        result = model.fit()
        B.append(result.params[1])

    B = np.array(B).reshape(-1, 1)
    s = dataset[x_col].var()

    # Set random seed
    np.random.seed(random_seed)

    # Generate random coefficients for w
    w = np.random.uniform(-1, 1, len(y_cols)).reshape(-1, 1)

    # Prepare data for gwas_indirect
    print(w.shape, P.shape, B.shape)
    data_xarray = build_xarray(w, P, [s], B.T, [[dataset.shape[0]]])

    # Run gwas_indirect
    BETA_indirect, SE_indirect, _, _ = gwas_indirect(data_xarray)

    # Run direct regression
    direct_params, direct_bse = compute_direct_regression(dataset, x_col, y_cols, w)

    # Compare coefficients and standard errors
    tolerance = 1e-6
    assert np.isclose(BETA_indirect[0], direct_params[1], rtol=tolerance)
    assert np.isclose(SE_indirect[0], direct_bse[1], rtol=tolerance)
