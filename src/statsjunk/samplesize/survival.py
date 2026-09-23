import numpy as np
from pydantic import BaseModel, Field


class SurvivalCriterion(BaseModel):
    """One row of the survival sample size calculation's criterion breakdown."""

    label: str
    sample_size: int
    shrinkage: float
    events: float
    epp: float


class SurvivalSampleSizeResult(BaseModel):
    """Minimum sample size for developing a survival outcome prediction model."""

    sample_size: int
    shrinkage: float
    parameters: int
    csrsquared: float = Field(ge=0, le=1)
    max_csrsquared: float = Field(ge=0, le=1)
    nagrsquared: float = Field(ge=0, le=1)
    events: float
    epp: float
    risk_at_timepoint: float = Field(ge=0, le=1)
    risk_low: float = Field(ge=0, le=1)
    risk_high: float = Field(ge=0, le=1)
    criteria: list[SurvivalCriterion]


def _max_csrsquared(rate: float, meanfup: float) -> float:
    """Maximum Cox-Snell R^2 attainable at this event rate and follow-up.

    Closed-form limit of the null-model log-likelihood per person-year
    (``d = rate * meanfup`` expected events per person), independent of
    sample size — the same derivation as `binary._max_csrsquared`, applied to
    a Poisson-type null model instead of a binomial one.
    """
    d = rate * meanfup
    ln_l_null_percapita = d * np.log(d) - d
    return float(1 - np.exp(2 * ln_l_null_percapita))


def compute_pmsampsize_survival(
    parameters: int,
    rate: float,
    timepoint: float,
    meanfup: float,
    csrsquared: float | None = None,
    nagrsquared: float | None = None,
    shrinkage: float = 0.9,
) -> SurvivalSampleSizeResult:
    """Minimum sample size for developing a survival outcome prediction model.

    Implements the criteria of Riley et al. 2019 ("Minimum sample size
    required for developing a multivariable prediction model: Part II binary
    and time-to-event outcomes"), analogous to the binary case but based on
    the expected number of events rather than the number of observations.

    Parameters
    ----------
    parameters : int
        Number of candidate predictor parameters for the new model.
    rate : float
        Overall event rate expected in the population, in the same time
        units as `meanfup` and `timepoint`.
    timepoint : float
        Timepoint of interest for prediction, same time units as `meanfup`.
    meanfup : float
        Average (mean) follow-up time anticipated in the development
        dataset, same time units as `timepoint`.
    csrsquared, nagrsquared : float, optional
        Exactly one must be given, as the anticipated Cox-Snell or
        Nagelkerke's R^2 of the new model — see `binary.compute_pmsampsize_binary`
        for how these relate.
    shrinkage : float, default=0.9
        Target shrinkage factor at internal validation, in (0, 1].

    Returns
    -------
    SurvivalSampleSizeResult
        `sample_size` (the maximum across all criteria) plus the full
        per-criterion breakdown and the estimated risk at `timepoint`.

    Raises
    ------
    ValueError
        If:
        - not exactly one of csrsquared/nagrsquared is given
        - parameters < 1
        - rate, timepoint, or meanfup is not positive
        - shrinkage is not in (0, 1]
        - the resulting csrsquared is <= 0, >= 1, or exceeds the maximum
          Cox-Snell R^2 attainable at this rate/follow-up
        - shrinkage is lower than csrsquared

    """
    if parameters < 1:
        raise ValueError("parameters must be at least 1.")
    if rate <= 0:
        raise ValueError("rate must be positive.")
    if timepoint <= 0:
        raise ValueError("timepoint must be positive.")
    if meanfup <= 0:
        raise ValueError("meanfup must be positive.")
    if not 0 < shrinkage <= 1:
        raise ValueError("shrinkage must be between 0 (exclusive) and 1.")

    given = [v is not None for v in (csrsquared, nagrsquared)]
    if sum(given) != 1:
        raise ValueError("Exactly one of csrsquared or nagrsquared must be given.")

    max_csrsquared = _max_csrsquared(rate, meanfup)

    if csrsquared is not None:
        r2a = csrsquared
    else:
        if not 0 < nagrsquared < 1:
            raise ValueError("nagrsquared must be between 0 and 1.")
        r2a = nagrsquared * max_csrsquared

    if not 0 < r2a < 1:
        raise ValueError(f"csrsquared must be between 0 and 1 (got {r2a}).")
    if shrinkage < r2a:
        raise ValueError(f"shrinkage ({shrinkage}) is lower than csrsquared ({r2a}).")
    if max_csrsquared < r2a:
        raise ValueError(
            f"csrsquared ({r2a}) is larger than the maximum Cox-Snell R^2 "
            f"attainable at this rate/follow-up ({max_csrsquared:.3f})."
        )

    nag_r2 = r2a / max_csrsquared

    # Criterion 1 - shrinkage.
    n1 = int(np.ceil(parameters / ((shrinkage - 1) * np.log(1 - (r2a / shrinkage)))))
    shrinkage_1 = shrinkage
    events_1 = n1 * rate * meanfup
    epp_1 = round(events_1 / parameters, 2)

    # Criterion 2 - small absolute difference in R^2 adjusted vs. apparent.
    s_4_small_diff = r2a / (r2a + (0.05 * max_csrsquared))
    n2 = int(
        np.ceil(
            parameters / ((s_4_small_diff - 1) * np.log(1 - (r2a / s_4_small_diff)))
        )
    )
    shrinkage_2 = s_4_small_diff
    events_2 = n2 * rate * meanfup
    epp_2 = round(events_2 / parameters, 2)

    # Criterion 3 - precise estimation of the risk at `timepoint`, evaluated
    # at n3 = max(n1, n2) rather than solved independently: once criteria 1
    # and 2 are met the event count is already large enough for this.
    n3 = max(n1, n2)
    se_rate = float(np.sqrt(rate / (meanfup * n3)))
    risk_high = 1 - np.exp(-(rate + 1.96 * se_rate) * timepoint)
    risk_low = 1 - np.exp(-(rate - 1.96 * se_rate) * timepoint)
    risk = 1 - np.exp(-rate * timepoint)
    events_3 = n3 * rate * meanfup
    epp_3 = round(events_3 / parameters, 2)

    shrinkage_3 = max(shrinkage, shrinkage_2)
    n_final = max(n1, n2, n3)
    shrinkage_final = shrinkage_3
    events_final = n_final * rate * meanfup
    epp_final = round(events_final / parameters, 2)

    criteria = [
        SurvivalCriterion(
            label="shrinkage",
            sample_size=n1,
            shrinkage=shrinkage_1,
            events=events_1,
            epp=epp_1,
        ),
        SurvivalCriterion(
            label="small_r2_difference",
            sample_size=n2,
            shrinkage=shrinkage_2,
            events=events_2,
            epp=epp_2,
        ),
        SurvivalCriterion(
            label="precise_risk_estimate",
            sample_size=n3,
            shrinkage=shrinkage_3,
            events=events_3,
            epp=epp_3,
        ),
    ]

    return SurvivalSampleSizeResult(
        sample_size=n_final,
        shrinkage=shrinkage_final,
        parameters=parameters,
        csrsquared=r2a,
        max_csrsquared=max_csrsquared,
        nagrsquared=nag_r2,
        events=events_final,
        epp=epp_final,
        risk_at_timepoint=float(risk),
        risk_low=float(risk_low),
        risk_high=float(risk_high),
        criteria=criteria,
    )
