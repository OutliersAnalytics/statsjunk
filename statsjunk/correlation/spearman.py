from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats

from .fisher import fisher_z, inverse_fisher_z


class SpearmanCorrelationResult(BaseModel):
    """Spearman rank correlation coefficient, its p-value, and confidence interval."""

    rho: float = Field(ge=-1, le=1)
    pvalue: float = Field(ge=0, le=1)
    low: float = Field(ge=-1, le=1)
    high: float = Field(ge=-1, le=1)


def compute_spearman_correlation(
    x: Sequence[float],
    y: Sequence[float],
    ci: float = 0.95,
) -> SpearmanCorrelationResult:
    """Compute Spearman's rank correlation coefficient and its confidence interval.

    What this solves
    -----------------
    Like Pearson correlation, this measures how strongly two variables move
    together — but based on their *ranks* rather than their raw values, so
    it still works when the relationship is consistently increasing or
    decreasing without being a straight line (e.g. "more of X always means
    more of Y, but not at a constant rate"), and it isn't thrown off by a
    few extreme outliers the way Pearson's r can be. Use this instead of
    `compute_pearson_correlation` when you care about "do they move in the
    same direction" more than "is the relationship linear."

    Parameters
    ----------
    x : Sequence[float]
        First variable.
    y : Sequence[float]
        Second variable.
    ci : float, default=0.95
        Confidence level for the correlation interval. Must be in (0, 1).

    Returns
    -------
    SpearmanCorrelationResult
        Spearman's rho, p-value, and confidence interval bounds.

    Raises
    ------
    ValueError
        If:
        - x and y have different lengths
        - fewer than 4 observations are provided
        - ci is not in (0, 1)
        - either input is constant

    References
    ----------
    - Spearman, C. (1904). "The proof and measurement of association
      between two things." American Journal of Psychology, 15(1), 72-101.
    - Fieller, E.C., Hartley, H.O., & Pearson, E.S. (1957). "Tests for rank
      correlation coefficients. I." Biometrika, 44(3/4), 470-481. (the 1.06
      correction to the Fisher z standard error used here)

    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length (got {len(x)} and {len(y)})"
        )
    if len(x) < 4:
        raise ValueError(
            "At least 4 observations are required to compute a confidence interval."
        )
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1.")
    if np.ptp(x) == 0:
        raise ValueError("x is constant.")
    if np.ptp(y) == 0:
        raise ValueError("y is constant.")

    result = stats.spearmanr(x, y)
    rho = float(result.statistic)
    pvalue = float(result.pvalue)

    if abs(rho) >= 1.0:
        return SpearmanCorrelationResult(rho=rho, pvalue=pvalue, low=rho, high=rho)

    n = len(x)
    z = fisher_z(rho)
    # Fieller, Hartley & Pearson (1957): Spearman's rho has ~1.06x the
    # sampling variance of Pearson's r for the same n, once z-transformed.
    se = float(np.sqrt(1.06 / (n - 3)))
    z_crit = float(stats.norm.ppf(0.5 + ci / 2))

    low = inverse_fisher_z(z - z_crit * se)
    high = inverse_fisher_z(z + z_crit * se)

    return SpearmanCorrelationResult(rho=rho, pvalue=pvalue, low=low, high=high)
