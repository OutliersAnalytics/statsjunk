import numpy as np
import pytest

from statsjunk.spatial import MoransIResult, compute_morans_i


def _grid(side: int) -> list[tuple[float, float]]:
    return [(float(i), float(j)) for i in range(side) for j in range(side)]


def test_clustered_values_give_positive_i():
    """A smooth gradient over a grid is strongly autocorrelated."""
    side = 12
    coords = _grid(side)
    values = [x + y for x, y in coords]

    result = compute_morans_i(values, coords, k=4)

    assert isinstance(result, MoransIResult)
    assert result.statistic > 0.5
    assert result.pvalue < 0.05
    assert result.n == side * side


def test_checkerboard_values_give_negative_i():
    side = 12
    coords = _grid(side)
    values = [(x + y) % 2 for x, y in coords]

    result = compute_morans_i(values, coords, k=4)

    assert result.statistic < 0
    assert result.pvalue < 0.05


def test_random_values_are_not_significant():
    rng = np.random.default_rng(42)
    coords = _grid(14)
    values = rng.normal(size=len(coords)).tolist()

    result = compute_morans_i(values, coords, k=8, seed=1)

    assert abs(result.statistic - result.expected) < 0.1
    assert result.pvalue > 0.05


def test_pvalue_is_reproducible():
    coords = _grid(10)
    values = [x * y for x, y in coords]

    a = compute_morans_i(values, coords, seed=7)
    b = compute_morans_i(values, coords, seed=7)

    assert a.pvalue == b.pvalue
    assert a.statistic == b.statistic


def test_raises_for_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        compute_morans_i([1, 2, 3], [(0, 0), (1, 1)])


def test_raises_for_too_few_regions():
    with pytest.raises(ValueError, match="At least 9 regions"):
        compute_morans_i([1, 2, 3], [(0, 0), (1, 1), (2, 2)], k=8)


def test_raises_for_constant_values():
    coords = _grid(6)
    with pytest.raises(ValueError, match="values is constant"):
        compute_morans_i([1.0] * len(coords), coords, k=4)


def test_raises_for_too_few_permutations():
    coords = _grid(6)
    values = [x + y for x, y in coords]
    with pytest.raises(ValueError, match="permutations must be at least 99"):
        compute_morans_i(values, coords, k=4, permutations=50)
