import numpy as np
import pytest
import scipy.stats

from indirect_gwas._igwas import compute_pvalue


@pytest.mark.parametrize(
    "t_stat, df",
    [
        (0.0, 1),
        (1.0, 1),
        (1.0, 2),
        (1e-4, 100_000),
        (1e-8, 100_000),
        (10, 100_000),
        (-10, 100_000),
        (20, 100_000),
        (50, 100_000),
        (100, 100_000),
        (500, 100_000),
        (1000, 100_000),
    ],
)
def test_compute_pvalue(t_stat, df):
    """
    Test that the p-value is computed correctly. All p-values are negative log10.
    """
    cpp_pvalue = compute_pvalue(t_stat, df)
    py_pvalue = -(scipy.stats.t(df=df).logsf(abs(t_stat)) + np.log(2)) / np.log(10)
    assert cpp_pvalue == pytest.approx(py_pvalue, rel=1e-6)
