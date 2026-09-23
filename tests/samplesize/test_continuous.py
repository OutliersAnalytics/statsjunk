import pytest

from statsjunk.samplesize.continuous import compute_pmsampsize_continuous


def test_matches_riley_worked_example():
    """Riley et al. 2019 (Part I) worked example: 25 parameters, R^2 of 0.2,
    intercept 1.9, sd 0.6 -> minimum n of 918."""
    result = compute_pmsampsize_continuous(
        parameters=25, rsquared=0.2, intercept=1.9, sd=0.6
    )

    assert result.sample_size == 918


def test_higher_rsquared_needs_fewer_observations():
    low = compute_pmsampsize_continuous(
        parameters=10, rsquared=0.2, intercept=1.0, sd=0.5
    )
    high = compute_pmsampsize_continuous(
        parameters=10, rsquared=0.6, intercept=1.0, sd=0.5
    )

    assert high.sample_size < low.sample_size


def test_intercept_ci_contains_intercept():
    result = compute_pmsampsize_continuous(
        parameters=25, rsquared=0.2, intercept=1.9, sd=0.6
    )

    assert result.intercept_low < result.intercept < result.intercept_high
    assert result.intercept_mmoe <= 1.1 + 1e-9


def test_raises_for_zero_intercept():
    with pytest.raises(ValueError, match="intercept must be nonzero"):
        compute_pmsampsize_continuous(parameters=25, rsquared=0.2, intercept=0, sd=0.6)


def test_raises_for_invalid_rsquared():
    with pytest.raises(ValueError, match="rsquared must be between 0 and 1"):
        compute_pmsampsize_continuous(
            parameters=25, rsquared=1.5, intercept=1.9, sd=0.6
        )


def test_raises_for_non_positive_sd():
    with pytest.raises(ValueError, match="sd must be positive"):
        compute_pmsampsize_continuous(parameters=25, rsquared=0.2, intercept=1.9, sd=0)


def test_raises_for_mmoe_not_greater_than_one():
    with pytest.raises(ValueError, match="mmoe must be greater than 1"):
        compute_pmsampsize_continuous(
            parameters=25, rsquared=0.2, intercept=1.9, sd=0.6, mmoe=1.0
        )
