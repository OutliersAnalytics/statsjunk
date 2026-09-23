from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats


class Coefficient(BaseModel):
    """One term of a fitted multiple linear regression.

    ``coef`` is on the regressor's own scale; ``coef_std`` is the standardized
    (beta) coefficient — the effect in standard deviations of ``y`` per standard
    deviation of the regressor — so magnitudes are comparable across regressors
    on different scales. Both are ``None`` for the intercept, as is ``vif``.
    """

    name: str
    coef: float
    coef_std: float | None
    std_error: float
    tvalue: float
    pvalue: float = Field(ge=0, le=1)
    ci_low: float
    ci_high: float
    vif: float | None = None


class MultipleRegressionResult(BaseModel):
    """Result of an ordinary-least-squares multiple linear regression."""

    coefficients: list[Coefficient]
    r_squared: float = Field(ge=0, le=1)
    adj_r_squared: float
    f_pvalue: float = Field(ge=0, le=1)
    n: int
    df_residual: int
    residuals: list[float]
    fitted: list[float]


def _variance_inflation_factors(x: np.ndarray) -> list[float]:
    """VIF for each column of ``x`` (an ``n x k`` matrix, no intercept column).

    ``VIF_j = 1 / (1 - R2_j)`` where ``R2_j`` comes from regressing column ``j``
    on the remaining columns plus an intercept. With a single regressor there is
    nothing to regress against and the VIF is 1.
    """
    n, k = x.shape
    factors: list[float] = []
    for j in range(k):
        others = np.delete(x, j, axis=1)
        design = np.column_stack([np.ones(n), others])
        target = x[:, j]
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        residuals = target - design @ beta
        ss_res = float(residuals @ residuals)
        ss_tot = float(((target - target.mean()) ** 2).sum())
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        factors.append(float("inf") if r_squared >= 1.0 else 1.0 / (1.0 - r_squared))
    return factors


def compute_multiple_regression(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    names: Sequence[str] | None = None,
    ci: float = 0.95,
) -> MultipleRegressionResult:
    """Fit a multiple linear regression by ordinary least squares.

    What this solves
    -----------------
    You have several variables you think together predict or explain an
    outcome (e.g. education, experience, and age predicting salary), and
    want to know each one's individual effect while accounting for the
    others. This fits the best-fitting linear combination, reports each
    predictor's effect size, standard error, and significance, and flags
    predictors that are too similar to each other to separate reliably
    (`vif`, variance inflation factor — a large value there is a warning
    sign that two or more of your predictors carry redundant information).

    Parameters
    ----------
    x : Sequence[Sequence[float]]
        Design matrix of shape ``(n_observations, n_regressors)`` — one row per
        observation, one column per independent variable (no intercept column).
    y : Sequence[float]
        Dependent variable, one value per observation.
    names : Sequence[str], optional
        Label for each regressor column; defaults to ``x1``, ``x2``, ...
    ci : float, default=0.95
        Confidence level for the per-coefficient intervals. Must be in (0, 1).

    Returns
    -------
    MultipleRegressionResult
        Per-coefficient estimates (raw and standardized), standard errors,
        two-sided Wald p-values, ``ci`` confidence intervals and VIFs — the
        intercept first, then one row per regressor — plus ``r_squared``,
        ``adj_r_squared``, the overall-model F-test ``f_pvalue``, ``n``,
        ``df_residual`` and the ``residuals``/``fitted`` vectors.

    Raises
    ------
    ValueError
        If:
        - x is not 2-D, or x and y disagree on the number of observations
        - there are no regressors, or fewer than ``k + 2`` observations
        - ci is not in (0, 1)
        - x or y contains a non-finite value
        - y or any regressor column is constant
        - the regressors are collinear (design matrix not full rank)
        - names is given with the wrong length

    References
    ----------
    - Draper, N.R. & Smith, H. (1998). Applied Regression Analysis (3rd
      ed.). Wiley.
    - Marquardt, D.W. (1970). "Generalized Inverses, Ridge Regression,
      Biased Linear Estimation, and Nonlinear Estimation." Technometrics,
      12(3), 591-612. (variance inflation factor)

    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 2:
        raise ValueError(
            "x must be a 2-D array of shape (n_observations, n_regressors)."
        )

    n, k = x.shape

    if len(y) != n:
        raise ValueError(
            f"x and y must have the same number of observations (got {n} and {len(y)})"
        )
    if k < 1:
        raise ValueError("At least one regressor is required.")
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1.")
    if n < k + 2:
        raise ValueError(
            f"At least {k + 2} observations are required for {k} regressor(s) "
            "(residual degrees of freedom must be at least 1)."
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must contain only finite values.")
    if np.ptp(y) == 0:
        raise ValueError("y is constant.")
    for j in range(k):
        if np.ptp(x[:, j]) == 0:
            raise ValueError(f"regressor at position {j} is constant.")

    if names is None:
        names = [f"x{j + 1}" for j in range(k)]
    elif len(names) != k:
        raise ValueError(
            f"names must have one entry per regressor (got {len(names)} for {k})"
        )

    design = np.column_stack([np.ones(n), x])

    if np.linalg.matrix_rank(design) < k + 1:
        raise ValueError("Regressors are collinear.")

    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    residuals = y - fitted

    df_residual = n - k - 1
    ss_res = float(residuals @ residuals)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / df_residual

    sigma_squared = ss_res / df_residual
    xtx_inv = np.linalg.inv(design.T @ design)
    std_errors = np.sqrt(np.diag(xtx_inv) * sigma_squared)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = np.where(std_errors > 0, beta / std_errors, 0.0)
    p_values = np.clip(2.0 * stats.t.sf(np.abs(t_values), df_residual), 0.0, 1.0)
    t_crit = float(stats.t.ppf((1 + ci) / 2, df_residual))
    margins = t_crit * std_errors

    if 1.0 - r_squared <= 0:
        f_pvalue = 0.0
    elif r_squared <= 0:
        f_pvalue = 1.0
    else:
        f_stat = (r_squared / k) / ((1.0 - r_squared) / df_residual)
        f_pvalue = float(stats.f.sf(f_stat, k, df_residual))

    y_std = float(y.std())
    x_stds = x.std(axis=0)
    vifs = _variance_inflation_factors(x)

    coefficients = [
        Coefficient(
            name="(intercept)",
            coef=float(beta[0]),
            coef_std=None,
            std_error=float(std_errors[0]),
            tvalue=float(t_values[0]),
            pvalue=float(p_values[0]),
            ci_low=float(beta[0] - margins[0]),
            ci_high=float(beta[0] + margins[0]),
            vif=None,
        )
    ]
    for j in range(k):
        coefficients.append(
            Coefficient(
                name=names[j],
                coef=float(beta[j + 1]),
                coef_std=float(beta[j + 1] * x_stds[j] / y_std),
                std_error=float(std_errors[j + 1]),
                tvalue=float(t_values[j + 1]),
                pvalue=float(p_values[j + 1]),
                ci_low=float(beta[j + 1] - margins[j + 1]),
                ci_high=float(beta[j + 1] + margins[j + 1]),
                vif=float(vifs[j]),
            )
        )

    return MultipleRegressionResult(
        coefficients=coefficients,
        r_squared=float(np.clip(r_squared, 0.0, 1.0)),
        adj_r_squared=float(adj_r_squared),
        f_pvalue=f_pvalue,
        n=n,
        df_residual=df_residual,
        residuals=[float(v) for v in residuals],
        fitted=[float(v) for v in fitted],
    )
