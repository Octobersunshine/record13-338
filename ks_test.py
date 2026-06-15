from scipy import stats
import numpy as np


def ks_two_sample(sample1, sample2):
    """
    Perform the two-sample Kolmogorov-Smirnov test.

    Parameters
    ----------
    sample1 : array-like
        First sample data.
    sample2 : array-like
        Second sample data.

    Returns
    -------
    dict
        {
            "D": float,  # KS D statistic (max absolute difference between CDFs)
            "p_value": float  # two-tailed p-value
        }

    Raises
    ------
    ValueError
        If either sample has fewer than 1 observation.
    TypeError
        If inputs cannot be converted to numeric arrays.
    """
    sample1 = np.asarray(sample1, dtype=float)
    sample2 = np.asarray(sample2, dtype=float)

    if sample1.size < 1:
        raise ValueError("sample1 must contain at least one observation")
    if sample2.size < 1:
        raise ValueError("sample2 must contain at least one observation")

    d_stat, p_val = stats.ks_2samp(sample1, sample2)

    return {"D": float(d_stat), "p_value": float(p_val)}


if __name__ == "__main__":
    np.random.seed(42)

    a = np.random.normal(loc=0, scale=1, size=500)
    b = np.random.normal(loc=0, scale=1, size=500)
    c = np.random.normal(loc=2, scale=1, size=500)

    result_same = ks_two_sample(a, b)
    print(f"Same distribution:  D={result_same['D']:.6f}, p={result_same['p_value']:.6f}")

    result_diff = ks_two_sample(a, c)
    print(f"Diff distribution:  D={result_diff['D']:.6f}, p={result_diff['p_value']:.6f}")
