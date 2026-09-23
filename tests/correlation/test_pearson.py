import numpy as np
import pytest

from statsjunk.correlation import (
    CorrelationResult,
    compute_pearson_correlation,
    compute_pearson_from_summary,
)


def test_compute_correlation_returns_result():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    result = compute_pearson_correlation(x, y)

    assert isinstance(result, CorrelationResult)
    assert result.r == pytest.approx(1.0)
    assert result.pvalue < 0.05
    assert result.low <= result.r <= result.high


def test_raises_for_different_lengths():
    with pytest.raises(ValueError, match="same length"):
        compute_pearson_correlation([1, 2, 3], [1, 2])


def test_raises_for_too_few_observations():
    with pytest.raises(ValueError, match="At least 4 observations"):
        compute_pearson_correlation([1, 2, 3], [4, 5, 6])


@pytest.mark.parametrize("ci", [0, 1, -0.1, 1.1])
def test_raises_for_invalid_ci(ci):
    with pytest.raises(ValueError, match="ci must be between 0 and 1"):
        compute_pearson_correlation([1, 2, 3, 4], [4, 5, 6, 7], ci=ci)


def test_raises_for_constant_x():
    with pytest.raises(ValueError, match="x is constant"):
        compute_pearson_correlation(
            [1, 1, 1, 1],
            [1, 2, 3, 4],
        )


def test_raises_for_constant_y():
    with pytest.raises(ValueError, match="y is constant"):
        compute_pearson_correlation(
            [1, 2, 3, 4],
            [5, 5, 5, 5],
        )


def test_correlation_is_inside_confidence_interval():
    rng = np.random.default_rng(42)

    x = rng.normal(size=100)
    y = 0.5 * x + rng.normal(size=100)

    result = compute_pearson_correlation(x, y)

    assert result.low < result.r < result.high


def test_higher_confidence_level_produces_wider_interval():
    rng = np.random.default_rng(42)

    x = rng.normal(size=100)
    y = 0.5 * x + rng.normal(size=100)

    ci95 = compute_pearson_correlation(x, y, ci=0.95)
    ci99 = compute_pearson_correlation(x, y, ci=0.99)

    width95 = ci95.high - ci95.low
    width99 = ci99.high - ci99.low

    assert width99 > width95


def test_from_summary_matches_full_computation():
    rng = np.random.default_rng(7)
    x = rng.normal(size=50)
    y = 0.6 * x + rng.normal(size=50)

    full = compute_pearson_correlation(x, y)
    from_summary = compute_pearson_from_summary(full.r, len(x))

    assert from_summary.r == pytest.approx(full.r)
    assert from_summary.pvalue == pytest.approx(full.pvalue, abs=1e-9)
    assert from_summary.low == pytest.approx(full.low, abs=1e-6)
    assert from_summary.high == pytest.approx(full.high, abs=1e-6)


def test_from_summary_returns_result():
    result = compute_pearson_from_summary(r=0.5, n=30)

    assert isinstance(result, CorrelationResult)
    assert result.r == 0.5
    assert result.low <= result.r <= result.high


def test_from_summary_perfect_correlation_has_zero_pvalue():
    result = compute_pearson_from_summary(r=1.0, n=10)

    assert result.pvalue == 0.0
    assert result.low == result.high == 1.0

    result = compute_pearson_from_summary(r=-1.0, n=10)

    assert result.pvalue == 0.0
    assert result.low == result.high == -1.0


def test_from_summary_raises_for_too_few_observations():
    with pytest.raises(ValueError, match="At least 4 observations"):
        compute_pearson_from_summary(r=0.5, n=3)


@pytest.mark.parametrize("r", [-1.5, 1.5])
def test_from_summary_raises_for_r_out_of_range(r):
    with pytest.raises(ValueError, match="r must be between -1 and 1"):
        compute_pearson_from_summary(r=r, n=30)


@pytest.mark.parametrize("ci", [0, 1, -0.1, 1.1])
def test_from_summary_raises_for_invalid_ci(ci):
    with pytest.raises(ValueError, match="ci must be between 0 and 1"):
        compute_pearson_from_summary(r=0.5, n=30, ci=ci)
