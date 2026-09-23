import numpy as np
from scipy import stats


def fisher_z(r: float) -> float:
    """Fisher z-transformation of a correlation coefficient.

    ``z = arctanh(r)``. Maps ``r`` from ``(-1, 1)`` to the whole real line,
    where it is approximately normally distributed — the basis for building
    confidence intervals and comparing correlations across samples.

    Parameters
    ----------
    r : float
        Correlation coefficient, in ``(-1, 1)``.

    Returns
    -------
    float
        The Fisher z-transformed value.

    """
    return float(np.arctanh(r))


def inverse_fisher_z(z: float) -> float:
    """Inverse Fisher z-transformation, back to a correlation coefficient.

    ``r = tanh(z)``. Inverse of `fisher_z`.

    Parameters
    ----------
    z : float
        A Fisher z-transformed value.

    Returns
    -------
    float
        The corresponding correlation coefficient, in ``(-1, 1)``.

    """
    return float(np.tanh(z))


def fisher_z_ci(r: float, n: int, ci: float = 0.95) -> tuple[float, float]:
    """Confidence interval for a correlation coefficient via the Fisher z-transform.

    Parameters
    ----------
    r : float
        Correlation coefficient, in ``(-1, 1)``. Callers should handle
        ``abs(r) >= 1`` themselves — the transform is undefined there.
    n : int
        Number of observations the correlation was computed over. Must be
        greater than 3.
    ci : float, default=0.95
        Confidence level. Must be in (0, 1).

    Returns
    -------
    tuple[float, float]
        ``(low, high)`` bounds of the confidence interval, in ``(-1, 1)``.

    Raises
    ------
    ValueError
        If n <= 3 or ci is not in (0, 1).

    """
    if n <= 3:
        raise ValueError("n must be greater than 3.")
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1.")

    z = fisher_z(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(0.5 + ci / 2)

    low = inverse_fisher_z(z - z_crit * se)
    high = inverse_fisher_z(z + z_crit * se)

    return low, high
