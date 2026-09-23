import pytest

from statsjunk.correlation.fisher import fisher_z, fisher_z_ci, inverse_fisher_z


def test_fisher_z_of_zero_is_zero():
    assert fisher_z(0.0) == pytest.approx(0.0)


def test_fisher_z_known_value():
    assert fisher_z(0.5) == pytest.approx(0.5493061443340549)


def test_inverse_fisher_z_undoes_fisher_z():
    for r in (-0.9, -0.5, 0.0, 0.3, 0.8):
        assert inverse_fisher_z(fisher_z(r)) == pytest.approx(r)


def test_fisher_z_ci_contains_r():
    low, high = fisher_z_ci(r=0.6, n=50)

    assert low < 0.6 < high


def test_fisher_z_ci_wider_for_higher_confidence():
    narrow = fisher_z_ci(r=0.6, n=50, ci=0.90)
    wide = fisher_z_ci(r=0.6, n=50, ci=0.99)

    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_fisher_z_ci_narrower_for_larger_n():
    small_n = fisher_z_ci(r=0.6, n=20)
    large_n = fisher_z_ci(r=0.6, n=500)

    assert (large_n[1] - large_n[0]) < (small_n[1] - small_n[0])


def test_fisher_z_ci_raises_for_n_too_small():
    with pytest.raises(ValueError, match="n must be greater than 3"):
        fisher_z_ci(r=0.5, n=3)


@pytest.mark.parametrize("ci", [0, 1, -0.1, 1.1])
def test_fisher_z_ci_raises_for_invalid_ci(ci):
    with pytest.raises(ValueError, match="ci must be between 0 and 1"):
        fisher_z_ci(r=0.5, n=30, ci=ci)
