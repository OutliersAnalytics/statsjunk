import pytest

from statsjunk.samplesize.binary import compute_pmsampsize_binary


def test_matches_riley_worked_example_with_csrsquared():
    """Riley et al. 2019 (Part II) worked example: 24 parameters, 17.4%
    prevalence, Cox-Snell R^2 of 0.288 -> minimum n of 662 (criterion 2)."""
    result = compute_pmsampsize_binary(
        parameters=24, prevalence=0.174, csrsquared=0.288, shrinkage=0.9
    )

    assert result.sample_size == 662
    assert result.criteria[0].sample_size == 623  # criterion 1 alone
    assert result.events == pytest.approx(115.188)
    assert result.epp == pytest.approx(4.8)


def test_cstatistic_approximates_equivalent_csrsquared():
    """The same worked example is also presented (Riley, Van Calster &
    Collins 2020) with an equivalent C-statistic of 0.89 instead of
    csrsquared=0.288 directly - the two should land close together."""
    direct = compute_pmsampsize_binary(
        parameters=24, prevalence=0.174, csrsquared=0.288, shrinkage=0.9
    )
    from_cstat = compute_pmsampsize_binary(
        parameters=24, prevalence=0.174, cstatistic=0.89, shrinkage=0.9
    )

    assert from_cstat.csrsquared == pytest.approx(direct.csrsquared, abs=0.02)
    assert from_cstat.sample_size == pytest.approx(direct.sample_size, rel=0.05)


def test_cstatistic_is_reproducible_for_same_seed():
    a = compute_pmsampsize_binary(
        parameters=24, prevalence=0.174, cstatistic=0.89, seed=7
    )
    b = compute_pmsampsize_binary(
        parameters=24, prevalence=0.174, cstatistic=0.89, seed=7
    )

    assert a.csrsquared == b.csrsquared
    assert a.sample_size == b.sample_size


def test_nagrsquared_is_equivalent_to_its_csrsquared():
    direct = compute_pmsampsize_binary(
        parameters=24, prevalence=0.174, csrsquared=0.288, shrinkage=0.9
    )
    from_nag = compute_pmsampsize_binary(
        parameters=24, prevalence=0.174, nagrsquared=direct.nagrsquared, shrinkage=0.9
    )

    assert from_nag.csrsquared == pytest.approx(direct.csrsquared)
    assert from_nag.sample_size == direct.sample_size


def test_raises_when_no_r2_source_given():
    with pytest.raises(ValueError, match="Exactly one of"):
        compute_pmsampsize_binary(parameters=24, prevalence=0.174)


def test_raises_when_multiple_r2_sources_given():
    with pytest.raises(ValueError, match="Exactly one of"):
        compute_pmsampsize_binary(
            parameters=24, prevalence=0.174, csrsquared=0.288, cstatistic=0.89
        )


def test_raises_for_prevalence_out_of_range():
    with pytest.raises(ValueError, match="prevalence must be between 0 and 1"):
        compute_pmsampsize_binary(parameters=24, prevalence=1.5, csrsquared=0.2)


def test_raises_when_shrinkage_lower_than_csrsquared():
    with pytest.raises(ValueError, match="shrinkage"):
        compute_pmsampsize_binary(
            parameters=24, prevalence=0.174, csrsquared=0.9, shrinkage=0.5
        )


def test_raises_when_csrsquared_exceeds_maximum():
    with pytest.raises(ValueError, match="maximum Cox-Snell"):
        compute_pmsampsize_binary(
            parameters=24, prevalence=0.174, csrsquared=0.99, shrinkage=1.0
        )


def test_raises_for_zero_parameters():
    with pytest.raises(ValueError, match="parameters must be at least 1"):
        compute_pmsampsize_binary(parameters=0, prevalence=0.174, csrsquared=0.2)
