import pytest

from statsjunk import PMSampleSize
from statsjunk.samplesize import (
    BinarySampleSizeResult,
    ContinuousSampleSizeResult,
    SurvivalSampleSizeResult,
)


def test_continuous_dispatch_matches_function():
    from statsjunk.samplesize.continuous import compute_continuous_sample_size

    direct = compute_continuous_sample_size(
        parameters=25, rsquared=0.2, intercept=1.9, sd=0.6
    )
    via_class = PMSampleSize(
        outcome_type="continuous", parameters=25, rsquared=0.2, intercept=1.9, sd=0.6
    ).compute()

    assert isinstance(via_class, ContinuousSampleSizeResult)
    assert via_class.sample_size == direct.sample_size


def test_binary_dispatch_matches_function():
    from statsjunk.samplesize.binary import compute_binary_sample_size

    direct = compute_binary_sample_size(
        parameters=24, prevalence=0.174, csrsquared=0.288
    )
    via_class = PMSampleSize(
        outcome_type="binary", parameters=24, prevalence=0.174, csrsquared=0.288
    ).compute()

    assert isinstance(via_class, BinarySampleSizeResult)
    assert via_class.sample_size == direct.sample_size


def test_survival_dispatch_matches_function():
    from statsjunk.samplesize.survival import compute_survival_sample_size

    direct = compute_survival_sample_size(
        parameters=30, csrsquared=0.051, rate=0.065, timepoint=2, meanfup=2.07
    )
    via_class = PMSampleSize(
        outcome_type="survival",
        parameters=30,
        csrsquared=0.051,
        rate=0.065,
        timepoint=2,
        meanfup=2.07,
    ).compute()

    assert isinstance(via_class, SurvivalSampleSizeResult)
    assert via_class.sample_size == direct.sample_size


def test_raises_for_missing_required_fields():
    with pytest.raises(ValueError, match="requires"):
        PMSampleSize(outcome_type="binary", parameters=24)
