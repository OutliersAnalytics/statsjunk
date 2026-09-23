from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats

from .fisher import fisher_z_ci


class CorrelationResult(BaseModel):
    r: float = Field(ge=-1, le=1)
    pvalue: float = Field(ge=0, le=1)
    low: float = Field(ge=-1, le=1)
    high: float = Field(ge=-1, le=1)


def compute_pearson_correlation(
    x: Sequence[float],
    y: Sequence[float],
    ci: float = 0.95,
) -> CorrelationResult:
    """Compute Pearson's correlation coefficient and its confidence interval.

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
    CorrelationResult
        Pearson correlation coefficient (`r`), p-value, and confidence
        interval bounds.

    Raises
    ------
    ValueError
        If:
        - x and y have different lengths
        - fewer than 4 observations are provided
        - ci is not in (0, 1)
        - either input is constant

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

    result = stats.pearsonr(x, y)
    ci_result = result.confidence_interval(confidence_level=ci)

    return CorrelationResult(
        r=result.statistic,
        pvalue=result.pvalue,
        low=ci_result.low,
        high=ci_result.high,
    )


def compute_pearson_from_summary(
    r: float,
    n: int,
    ci: float = 0.95,
) -> CorrelationResult:
    """Derive a Pearson correlation's p-value and confidence interval from
    summary statistics (r, n) alone, without the underlying arrays.

    For callers that already computed r elsewhere over many groups in one
    pass (e.g. a SQL `corr()` aggregate grouped by category) and would
    otherwise have to re-fetch every group's raw per-observation arrays just
    to get significance out of `compute_pearson_correlation`.

    Parameters
    ----------
    r : float
        Pearson correlation coefficient, in [-1, 1].
    n : int
        Number of observations the correlation was computed over.
    ci : float, default=0.95
        Confidence level for the correlation interval. Must be in (0, 1).

    Returns
    -------
    CorrelationResult
        Same shape as `compute_pearson_correlation`.

    Raises
    ------
    ValueError
        If n < 4, r is outside [-1, 1], or ci is not in (0, 1).

    """
    if n < 4:
        raise ValueError(
            "At least 4 observations are required to compute a confidence interval."
        )
    if not -1 <= r <= 1:
        raise ValueError(f"r must be between -1 and 1 (got {r}).")
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1.")

    if abs(r) >= 1.0:
        return CorrelationResult(r=r, pvalue=0.0, low=r, high=r)

    df = n - 2
    t_stat = r * np.sqrt(df / (1 - r**2))
    pvalue = float(2 * stats.t.sf(abs(t_stat), df))

    low, high = fisher_z_ci(r, n, ci)

    return CorrelationResult(r=r, pvalue=pvalue, low=low, high=high)
