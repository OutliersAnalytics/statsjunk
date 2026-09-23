from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats


class RegressionResult(BaseModel):
    """Result of a simple linear regression, with slope diagnostics."""

    slope: float
    intercept: float
    r_squared: float = Field(ge=0, le=1)
    slope_pvalue: float = Field(ge=0, le=1)
    slope_low: float
    slope_high: float


def compute_linear_regression(
    x: Sequence[float],
    y: Sequence[float],
    ci: float = 0.95,
) -> RegressionResult:
    """Fit a simple linear regression model and its slope diagnostics.

    Parameters
    ----------
    x : Sequence[float]
        Independent variable.
    y : Sequence[float]
        Dependent variable.
    ci : float, default=0.95
        Confidence level for the slope interval. Must be in (0, 1).

    Returns
    -------
    RegressionResult
        Fitted ``slope`` and ``intercept``; ``r_squared`` (share of the
        variance in ``y`` explained by the fit); ``slope_pvalue`` (two-sided
        Wald test of ``slope == 0`` — identical to Pearson's p-value in a
        simple regression); and ``slope_low``/``slope_high``, the ``ci``
        confidence interval for the slope.

    Raises
    ------
    ValueError
        If:
        - x and y have different lengths
        - fewer than 3 observations are provided
        - ci is not in (0, 1)
        - either input is constant

    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length (got {len(x)} and {len(y)})"
        )
    if len(x) < 3:
        raise ValueError(
            "At least 3 observations are required to compute slope diagnostics."
        )
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1.")
    if np.ptp(x) == 0:
        raise ValueError("x is constant.")
    if np.ptp(y) == 0:
        raise ValueError("y is constant.")

    result = stats.linregress(x, y)

    df = len(x) - 2
    t_crit = float(stats.t.ppf((1 + ci) / 2, df))
    margin = t_crit * result.stderr

    return RegressionResult(
        slope=float(result.slope),
        intercept=float(result.intercept),
        r_squared=float(result.rvalue) ** 2,
        slope_pvalue=float(result.pvalue),
        slope_low=float(result.slope - margin),
        slope_high=float(result.slope + margin),
    )
