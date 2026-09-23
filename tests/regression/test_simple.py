import pytest

from statsjunk.regression import compute_linear_regression


def test_returns_expected_line():
    result = compute_linear_regression(
        [1, 2, 3, 4],
        [2, 4, 6, 8],
    )

    assert result.slope == pytest.approx(2.0)
    assert result.intercept == pytest.approx(0.0)


def test_perfect_fit_diagnostics():
    result = compute_linear_regression(
        [1, 2, 3, 4],
        [2, 4, 6, 8],
    )

    assert result.r_squared == pytest.approx(1.0)
    assert result.slope_pvalue == pytest.approx(0.0, abs=1e-9)
    assert result.slope_low == pytest.approx(2.0)
    assert result.slope_high == pytest.approx(2.0)


def test_noisy_fit_diagnostics():
    result = compute_linear_regression(
        [1, 2, 3, 4, 5, 6],
        [2, 1, 4, 3, 6, 5],
    )

    assert 0 < result.r_squared < 1
    assert 0 < result.slope_pvalue < 1
    assert result.slope_low < result.slope < result.slope_high


def test_wider_ci_is_wider():
    args = ([1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5])
    narrow = compute_linear_regression(*args, ci=0.90)
    wide = compute_linear_regression(*args, ci=0.99)

    assert (wide.slope_high - wide.slope_low) > (narrow.slope_high - narrow.slope_low)


def test_raises_for_different_lengths():
    with pytest.raises(ValueError, match="same length"):
        compute_linear_regression([1, 2, 3], [1, 2])


def test_raises_for_too_few_observations():
    with pytest.raises(ValueError, match="At least 3 observations"):
        compute_linear_regression([1, 2], [2, 4])


def test_raises_for_invalid_ci():
    with pytest.raises(ValueError, match="ci must be between 0 and 1"):
        compute_linear_regression([1, 2, 3, 4], [2, 4, 6, 8], ci=1.5)


def test_raises_for_constant_x():
    with pytest.raises(ValueError, match="x is constant"):
        compute_linear_regression(
            [1, 1, 1, 1],
            [2, 3, 4, 5],
        )


def test_raises_for_constant_y():
    with pytest.raises(ValueError, match="y is constant"):
        compute_linear_regression(
            [1, 2, 3, 4],
            [5, 5, 5, 5],
        )


def test_fits_line_with_nonzero_intercept():
    result = compute_linear_regression(
        [1, 2, 3, 4],
        [3, 5, 7, 9],
    )

    assert result.slope == pytest.approx(2.0)
    assert result.intercept == pytest.approx(1.0)
