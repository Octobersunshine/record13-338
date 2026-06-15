from scipy import stats
import numpy as np


def _vargha_delaney_a(sample1, sample2):
    """
    Compute Vargha-Delaney A statistic (probability of superiority).

    A = 0.5 means the two samples are indistinguishable in rank.
    A > 0.5 means sample1 tends to have larger values.
    A < 0.5 means sample2 tends to have larger values.
    """
    n1 = len(sample1)
    n2 = len(sample2)
    combined = np.concatenate([sample1, sample2])
    ranks = stats.rankdata(combined)
    r1 = np.sum(ranks[:n1])
    a = (r1 / n1 - (n1 + 1) / 2) / n2
    return a


def _interpret_d(d):
    """
    Qualitative interpretation of KS D statistic as effect size.

    Based on common conventions for two-sample KS test:
    - D < 0.1 : negligible
    - 0.1 <= D < 0.3 : small
    - 0.3 <= D < 0.5 : medium
    - D >= 0.5 : large
    """
    if d < 0.1:
        return "negligible"
    elif d < 0.3:
        return "small"
    elif d < 0.5:
        return "medium"
    else:
        return "large"


def _compute_ecdf_data(sample1, sample2, d_stat):
    """
    Compute empirical CDF data for both samples on a common x grid.

    Returns
    -------
    dict
        {
            "x": list of x-axis values (sorted unique combined data)
            "sample1_cdf": list of CDF values for sample1 at each x
            "sample2_cdf": list of CDF values for sample2 at each x
            "d_max_point": {"x", "cdf1", "cdf2", "diff"} where |cdf1-cdf2| == D
        }
    """
    s1 = np.sort(sample1)
    s2 = np.sort(sample2)
    n1 = len(s1)
    n2 = len(s2)

    x = np.unique(np.concatenate([s1, s2]))

    cdf1 = np.searchsorted(s1, x, side='right') / n1
    cdf2 = np.searchsorted(s2, x, side='right') / n2

    diffs = np.abs(cdf1 - cdf2)
    max_idx = np.argmax(diffs)

    return {
        "x": x.tolist(),
        "sample1_cdf": cdf1.tolist(),
        "sample2_cdf": cdf2.tolist(),
        "d_max_point": {
            "x": float(x[max_idx]),
            "cdf1": float(cdf1[max_idx]),
            "cdf2": float(cdf2[max_idx]),
            "diff": float(diffs[max_idx])
        }
    }


def ks_two_sample(sample1, sample2):
    """
    Perform the two-sample Kolmogorov-Smirnov test with effect size and CDF data.

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
            "D": float,                 # KS D statistic
            "p_value": float,           # two-tailed p-value
            "n1": int,                  # sample size of sample1
            "n2": int,                  # sample size of sample2
            "effect_size": {
                "D": float,             # KS D statistic (also effect size)
                "D_magnitude": str,     # qualitative magnitude of D
                "VarghaDelaney_A": float  # probability of superiority
            },
            "cdf_data": {
                "x": list[float],       # common x-axis values
                "sample1_cdf": list[float],  # CDF of sample1 at each x
                "sample2_cdf": list[float],  # CDF of sample2 at each x
                "d_max_point": {        # where D occurs
                    "x": float,
                    "cdf1": float,
                    "cdf2": float,
                    "diff": float
                }
            }
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

    n1 = sample1.size
    n2 = sample2.size

    if n1 < 1:
        raise ValueError("sample1 must contain at least one observation")
    if n2 < 1:
        raise ValueError("sample2 must contain at least one observation")

    d_stat, p_val = stats.ks_2samp(sample1, sample2)
    vda = _vargha_delaney_a(sample1, sample2)
    ecdf_data = _compute_ecdf_data(sample1, sample2, d_stat)

    return {
        "D": float(d_stat),
        "p_value": float(p_val),
        "n1": int(n1),
        "n2": int(n2),
        "effect_size": {
            "D": float(d_stat),
            "D_magnitude": _interpret_d(d_stat),
            "VarghaDelaney_A": float(vda)
        },
        "cdf_data": ecdf_data
    }


if __name__ == "__main__":
    np.random.seed(42)

    a = np.random.normal(loc=0, scale=1, size=500)
    b = np.random.normal(loc=0, scale=1, size=500)
    c = np.random.normal(loc=2, scale=1, size=500)

    result_same = ks_two_sample(a, b)
    print(f"Same distribution (n1={result_same['n1']}, n2={result_same['n2']}):")
    print(f"  D={result_same['D']:.6f}, p={result_same['p_value']:.6f}")
    print(f"  effect size: D_magnitude={result_same['effect_size']['D_magnitude']}, "
          f"Vargha-Delaney A={result_same['effect_size']['VarghaDelaney_A']:.4f}")
    dmp = result_same['cdf_data']['d_max_point']
    print(f"  CDF data: {len(result_same['cdf_data']['x'])} points, "
          f"D at x={dmp['x']:.4f}, |cdf1-cdf2|={dmp['diff']:.6f}")
    print()

    result_diff = ks_two_sample(a, c)
    print(f"Diff distribution (n1={result_diff['n1']}, n2={result_diff['n2']}):")
    print(f"  D={result_diff['D']:.6f}, p={result_diff['p_value']:.6f}")
    print(f"  effect size: D_magnitude={result_diff['effect_size']['D_magnitude']}, "
          f"Vargha-Delaney A={result_diff['effect_size']['VarghaDelaney_A']:.4f}")
    dmp = result_diff['cdf_data']['d_max_point']
    print(f"  CDF data: {len(result_diff['cdf_data']['x'])} points, "
          f"D at x={dmp['x']:.4f}, cdf1={dmp['cdf1']:.4f}, cdf2={dmp['cdf2']:.4f}")
    print()

    big_a = np.random.normal(loc=0, scale=1, size=100000)
    big_b = np.random.normal(loc=0.02, scale=1, size=100000)
    result_big = ks_two_sample(big_a, big_b)
    print(f"Large sample, tiny diff (n1={result_big['n1']}, n2={result_big['n2']}):")
    print(f"  D={result_big['D']:.6f}, p={result_big['p_value']:.6e}")
    print(f"  effect size: D_magnitude={result_big['effect_size']['D_magnitude']}, "
          f"Vargha-Delaney A={result_big['effect_size']['VarghaDelaney_A']:.4f}")
    dmp = result_big['cdf_data']['d_max_point']
    print(f"  CDF data: {len(result_big['cdf_data']['x'])} points, "
          f"D at x={dmp['x']:.4f}, |cdf1-cdf2|={dmp['diff']:.6f}")
    print("  -> p-value is tiny but effect size is negligible: "
          "statistically significant, practically irrelevant.")
