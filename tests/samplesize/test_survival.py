import pytest

from statsjunk.samplesize.survival import compute_survival_sample_size


def test_matches_riley_worked_example():
    """Riley et al. 2020 (BMJ) worked example: 30 parameters, Cox-Snell R^2
    of 0.051, event rate 0.065, 2.07 years mean follow-up, 2-year timepoint
    -> minimum n of 5143."""
    result = compute_survival_sample_size(
        parameters=30, csrsquared=0.051, rate=0.065, timepoint=2, meanfup=2.07
    )

    assert result.sample_size == 5143


def test_small_example():
    result = compute_survival_sample_size(
        parameters=5, csrsquared=0.1, rate=0.1, timepoint=5, meanfup=10, shrinkage=0.9
    )

    assert result.sample_size == 425


def test_nagrsquared_is_equivalent_to_its_csrsquared():
    direct = compute_survival_sample_size(
        parameters=30, csrsquared=0.051, rate=0.065, timepoint=2, meanfup=2.07
    )
    from_nag = compute_survival_sample_size(
        parameters=30,
        nagrsquared=direct.nagrsquared,
        rate=0.065,
        timepoint=2,
        meanfup=2.07,
    )

    assert from_nag.csrsquared == pytest.approx(direct.csrsquared)
    assert from_nag.sample_size == direct.sample_size


def test_risk_at_timepoint_is_between_bounds():
    result = compute_survival_sample_size(
        parameters=30, csrsquared=0.051, rate=0.065, timepoint=2, meanfup=2.07
    )

    assert result.risk_low < result.risk_at_timepoint < result.risk_high


def test_raises_when_no_r2_source_given():
    with pytest.raises(ValueError, match="Exactly one of"):
        compute_survival_sample_size(
            parameters=30, rate=0.065, timepoint=2, meanfup=2.07
        )


def test_raises_for_non_positive_rate():
    with pytest.raises(ValueError, match="rate must be positive"):
        compute_survival_sample_size(
            parameters=30, csrsquared=0.051, rate=0, timepoint=2, meanfup=2.07
        )


def test_raises_when_csrsquared_exceeds_maximum():
    with pytest.raises(ValueError, match="maximum Cox-Snell"):
        compute_survival_sample_size(
            parameters=30,
            csrsquared=0.99,
            rate=0.065,
            timepoint=2,
            meanfup=2.07,
            shrinkage=1.0,
        )
