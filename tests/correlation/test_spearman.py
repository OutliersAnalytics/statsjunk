import numpy as np
import pytest

from statsjunk.correlation.spearman import (
    SpearmanCorrelationResult,
    compute_spearman_correlation,
)


def test_perfect_monotonic_nonlinear_relationship_is_one():
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]  # y = x^2: monotonic but not linear

    result = compute_spearman_correlation(x, y)

    assert isinstance(result, SpearmanCorrelationResult)
    assert result.rho == pytest.approx(1.0)
    assert result.low == pytest.approx(1.0)
    assert result.high == pytest.approx(1.0)


def test_perfect_decreasing_is_minus_one():
    result = compute_spearman_correlation([1, 2, 3, 4], [8, 6, 4, 2])

    assert result.rho == pytest.approx(-1.0)


def test_raises_for_different_lengths():
    with pytest.raises(ValueError, match="same length"):
        compute_spearman_correlation([1, 2, 3], [1, 2])


def test_raises_for_too_few_observations():
    with pytest.raises(ValueError, match="At least 4 observations"):
        compute_spearman_correlation([1, 2, 3], [4, 5, 6])


@pytest.mark.parametrize("ci", [0, 1, -0.1, 1.1])
def test_raises_for_invalid_ci(ci):
    with pytest.raises(ValueError, match="ci must be between 0 and 1"):
        compute_spearman_correlation([1, 2, 3, 4], [4, 5, 6, 7], ci=ci)


def test_raises_for_constant_x():
    with pytest.raises(ValueError, match="x is constant"):
        compute_spearman_correlation([1, 1, 1, 1], [1, 2, 3, 4])


def test_raises_for_constant_y():
    with pytest.raises(ValueError, match="y is constant"):
        compute_spearman_correlation([1, 2, 3, 4], [5, 5, 5, 5])


def test_confidence_interval_contains_rho():
    rng = np.random.default_rng(42)
    x = rng.normal(size=100)
    y = x**3 + rng.normal(scale=0.5, size=100)

    result = compute_spearman_correlation(x, y)

    assert result.low < result.rho < result.high


def test_higher_confidence_level_produces_wider_interval():
    rng = np.random.default_rng(42)
    x = rng.normal(size=100)
    y = x**3 + rng.normal(scale=0.5, size=100)

    ci95 = compute_spearman_correlation(x, y, ci=0.95)
    ci99 = compute_spearman_correlation(x, y, ci=0.99)

    assert (ci99.high - ci99.low) > (ci95.high - ci95.low)


def test_wider_than_pearson_ci_for_same_data():
    """Spearman's z-standard-error carries the 1.06 correction, so its
    interval should be at least as wide as an equivalent Pearson interval
    computed on the same rho/n."""
    from statsjunk.correlation.fisher import fisher_z_ci

    rng = np.random.default_rng(7)
    x = rng.normal(size=60)
    y = x**3 + rng.normal(scale=0.5, size=60)

    spearman = compute_spearman_correlation(x, y)
    pearson_style_low, pearson_style_high = fisher_z_ci(spearman.rho, len(x))

    assert (spearman.high - spearman.low) > (pearson_style_high - pearson_style_low)


def test_uncorrelated_variables_have_small_rho():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = rng.normal(size=200)

    result = compute_spearman_correlation(x, y)

    assert abs(result.rho) < 0.2
    assert result.pvalue > 0.01
