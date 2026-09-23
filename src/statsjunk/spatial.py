from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field
from scipy import sparse
from scipy.spatial import cKDTree


class MoransIResult(BaseModel):
    """Global Moran's I spatial autocorrelation statistic."""

    statistic: float
    expected: float
    pvalue: float = Field(ge=0, le=1)
    n: int


def compute_morans_i(
    values: Sequence[float],
    coords: Sequence[Sequence[float]],
    k: int = 8,
    permutations: int = 999,
    seed: int = 0,
) -> MoransIResult:
    """Global Moran's I with k-nearest-neighbour spatial weights.

    Answers "are nearby regions more alike than distant ones?" — used as a
    caveat on a bivariate correlation, since strong spatial autocorrelation
    means the observations are not independent and the correlation's
    effective sample size is smaller than n.

    Parameters
    ----------
    values : Sequence[float]
        The variable of interest, one value per region.
    coords : Sequence[Sequence[float]]
        Region centroid coordinates as ``(x, y)`` pairs, same order as
        ``values``. Lon/lat in degrees is fine for this diagnostic.
    k : int, default=8
        Number of nearest neighbours each region is weighted against. Weights
        are row-standardised (each neighbour counts ``1 / k``).
    permutations : int, default=999
        Random permutations for the pseudo p-value.
    seed : int, default=0
        Seed for the permutation RNG, so the p-value is reproducible.

    Returns
    -------
    MoransIResult
        ``statistic`` (Moran's I, ~-1..1), ``expected`` (``-1 / (n - 1)``, the
        value under no autocorrelation), ``pvalue`` (one-sided permutation
        p-value in the direction of the observed statistic), and ``n``.

    Raises
    ------
    ValueError
        If:
        - values and coords have different lengths
        - k < 1
        - permutations < 99
        - fewer than k + 1 regions are provided
        - values is constant

    """
    values = np.asarray(values, dtype=float)
    coords = np.asarray(coords, dtype=float)

    if len(values) != len(coords):
        raise ValueError(
            "values and coords must have the same length "
            f"(got {len(values)} and {len(coords)})"
        )
    if k < 1:
        raise ValueError("k must be at least 1.")
    if permutations < 99:
        raise ValueError("permutations must be at least 99.")
    n = len(values)
    if n < k + 1:
        raise ValueError(f"At least {k + 1} regions are required for k = {k}.")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be a sequence of (x, y) pairs.")
    if np.ptp(values) == 0:
        raise ValueError("values is constant.")

    # k nearest neighbours, dropping each point's match with itself.
    _, idx = cKDTree(coords).query(coords, k=k + 1)
    neighbours = idx[:, 1:]

    rows = np.repeat(np.arange(n), k)
    weights = sparse.csr_matrix(
        (np.full(n * k, 1.0 / k), (rows, neighbours.ravel())),
        shape=(n, n),
    )

    z = values - values.mean()
    denominator = float(z @ z)

    def morans(v: np.ndarray) -> float:
        return float(v @ weights.dot(v) / denominator)

    statistic = morans(z)

    rng = np.random.default_rng(seed)
    simulated = np.array([morans(rng.permutation(z)) for _ in range(permutations)])

    expected = -1.0 / (n - 1)
    if statistic >= expected:
        hits = int(np.sum(simulated >= statistic))
    else:
        hits = int(np.sum(simulated <= statistic))
    pvalue = (hits + 1) / (permutations + 1)

    return MoransIResult(
        statistic=statistic,
        expected=expected,
        pvalue=pvalue,
        n=n,
    )
