import numpy as np
import pytest

from statsjunk.regression.multiple import compute_multiple_regression


def _named(result):
    return {c.name: c for c in result.coefficients}


def test_recovers_known_coefficients():
    # y = 1 + 2*x1 - 3*x2, no noise
    x1 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    x2 = [1.0, 0.0, 2.0, 1.0, 3.0, 0.0]
    y = [1 + 2 * a - 3 * b for a, b in zip(x1, x2)]

    result = compute_multiple_regression(list(zip(x1, x2)), y, names=["x1", "x2"])
    coefs = _named(result)

    assert coefs["(intercept)"].coef == pytest.approx(1.0)
    assert coefs["x1"].coef == pytest.approx(2.0)
    assert coefs["x2"].coef == pytest.approx(-3.0)
    assert result.n == 6
    assert result.df_residual == 3


def test_perfect_fit_diagnostics():
    x1 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    x2 = [1.0, 0.0, 2.0, 1.0, 3.0, 0.0]
    y = [1 + 2 * a - 3 * b for a, b in zip(x1, x2)]

    result = compute_multiple_regression(list(zip(x1, x2)), y)

    assert result.r_squared == pytest.approx(1.0)
    assert result.adj_r_squared == pytest.approx(1.0)
    assert result.f_pvalue == pytest.approx(0.0, abs=1e-9)
    for coef in result.coefficients:
        assert coef.ci_low == pytest.approx(coef.coef)
        assert coef.ci_high == pytest.approx(coef.coef)
    assert np.allclose(result.residuals, 0.0)


def test_noisy_fit_diagnostics():
    rng = np.random.default_rng(42)
    n = 60
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 3.0 * x2 + rng.normal(scale=0.5, size=n)

    result = compute_multiple_regression(
        np.column_stack([x1, x2]), y, names=["x1", "x2"]
    )
    coefs = _named(result)

    assert 0 < result.r_squared < 1
    assert result.adj_r_squared < result.r_squared
    assert coefs["x1"].coef == pytest.approx(2.0, abs=0.3)
    assert coefs["x2"].coef == pytest.approx(-3.0, abs=0.3)
    for name in ("x1", "x2"):
        assert coefs[name].ci_low < coefs[name].coef < coefs[name].ci_high
        assert 0 <= coefs[name].pvalue <= 1
    # both slopes clearly non-zero => model is significant
    assert result.f_pvalue < 0.01


def test_standardized_coefficient_is_comparable_across_scales():
    rng = np.random.default_rng(0)
    n = 80
    small = rng.normal(scale=1.0, size=n)
    large = rng.normal(scale=1000.0, size=n)
    # equal standardized effect, wildly different raw scale
    y = (
        5.0 * (small / small.std())
        + 5.0 * (large / large.std())
        + rng.normal(scale=0.1, size=n)
    )

    result = compute_multiple_regression(
        np.column_stack([small, large]), y, names=["small", "large"]
    )
    coefs = _named(result)

    assert abs(coefs["small"].coef) > 100 * abs(coefs["large"].coef)
    assert coefs["small"].coef_std == pytest.approx(coefs["large"].coef_std, rel=0.1)
    assert coefs["(intercept)"].coef_std is None


def test_vif_flags_correlated_regressors():
    rng = np.random.default_rng(7)
    n = 50
    x1 = rng.normal(size=n)
    x2 = x1 + rng.normal(scale=0.05, size=n)  # almost a copy of x1
    x3 = rng.normal(size=n)
    y = x1 + x3 + rng.normal(scale=0.5, size=n)

    result = compute_multiple_regression(
        np.column_stack([x1, x2, x3]), y, names=["x1", "x2", "x3"]
    )
    coefs = _named(result)

    assert coefs["x1"].vif > 10
    assert coefs["x2"].vif > 10
    assert coefs["x3"].vif < 5
    assert coefs["(intercept)"].vif is None


def test_single_regressor_has_vif_one():
    result = compute_multiple_regression([[1.0], [2.0], [3.0], [4.0]], [2, 3, 5, 4])

    assert _named(result)["x1"].vif == pytest.approx(1.0)


def test_wider_ci_is_wider():
    x = np.column_stack([[1.0, 2, 3, 4, 5, 6], [2.0, 1, 4, 3, 6, 5]])
    y = [2.0, 1, 4, 3, 6, 6]
    narrow = compute_multiple_regression(x, y, ci=0.90)
    wide = compute_multiple_regression(x, y, ci=0.99)

    n_width = narrow.coefficients[1].ci_high - narrow.coefficients[1].ci_low
    w_width = wide.coefficients[1].ci_high - wide.coefficients[1].ci_low
    assert w_width > n_width


def test_raises_for_mismatched_observations():
    with pytest.raises(ValueError, match="same number of observations"):
        compute_multiple_regression([[1, 2], [3, 4], [5, 6]], [1, 2])


def test_raises_for_no_regressors():
    with pytest.raises(ValueError, match="At least one regressor"):
        compute_multiple_regression(np.empty((5, 0)), [1, 2, 3, 4, 5])


def test_raises_for_too_few_observations():
    with pytest.raises(ValueError, match="At least 4 observations"):
        compute_multiple_regression([[1, 1], [2, 3], [3, 2]], [1, 2, 3])


def test_raises_for_invalid_ci():
    with pytest.raises(ValueError, match="ci must be between 0 and 1"):
        compute_multiple_regression([[1], [2], [3], [4]], [1, 2, 3, 4], ci=1.5)


def test_raises_for_constant_y():
    with pytest.raises(ValueError, match="y is constant"):
        compute_multiple_regression([[1], [2], [3], [4]], [5, 5, 5, 5])


def test_raises_for_constant_regressor():
    with pytest.raises(ValueError, match="regressor at position 1 is constant"):
        compute_multiple_regression([[1, 7], [2, 7], [3, 7], [4, 7]], [1, 2, 3, 5])


def test_raises_for_collinear_regressors():
    # x3 = x1 + x2 exactly
    x1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    x2 = [2.0, 1.0, 4.0, 3.0, 6.0]
    x3 = [a + b for a, b in zip(x1, x2)]
    with pytest.raises(ValueError, match="collinear"):
        compute_multiple_regression(
            list(zip(x1, x2, x3)), [1, 2, 3, 4, 5], names=["x1", "x2", "x3"]
        )


def test_raises_for_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        compute_multiple_regression([[1], [2], [3], [float("nan")]], [1, 2, 3, 4])


def test_raises_for_wrong_names_length():
    with pytest.raises(ValueError, match="one entry per regressor"):
        compute_multiple_regression(
            [[1, 2], [3, 4], [5, 6], [7, 8]], [1, 2, 3, 4], names=["only-one"]
        )
