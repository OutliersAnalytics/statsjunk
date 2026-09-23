import numpy as np
from pydantic import BaseModel, Field
from scipy import optimize, stats


class BinaryCriterion(BaseModel):
    """One row of the binary sample size calculation's criterion breakdown."""

    label: str
    sample_size: int
    shrinkage: float
    events: float
    epp: float


class BinarySampleSizeResult(BaseModel):
    """Minimum sample size for developing a binary outcome prediction model."""

    sample_size: int
    shrinkage: float
    parameters: int
    csrsquared: float = Field(ge=0, le=1)
    max_csrsquared: float = Field(ge=0, le=1)
    nagrsquared: float = Field(ge=0, le=1)
    prevalence: float = Field(gt=0, lt=1)
    events: float
    epp: float
    criteria: list[BinaryCriterion]


def _max_csrsquared(prevalence: float) -> float:
    """Maximum Cox-Snell R^2 attainable for a binary outcome at this prevalence.

    Derived from the per-observation null-model log-likelihood
    (``prevalence*log(prevalence) + (1-prevalence)*log(1-prevalence)``), which
    is independent of sample size (Riley et al. 2019, Part II, eq. 23).
    """
    ln_l_null_percapita = prevalence * np.log(prevalence) + (1 - prevalence) * np.log(
        1 - prevalence
    )
    return float(1 - np.exp(2 * ln_l_null_percapita))


def _fit_logistic_1d(lp: np.ndarray, y: np.ndarray) -> float:
    """Fit y ~ logistic(a + b*lp) by maximum likelihood, return the log-likelihood."""

    def neg_log_lik(params: np.ndarray) -> float:
        a, b = params
        eta = a + b * lp
        return float(-np.sum(y * eta - np.logaddexp(0, eta)))

    result = optimize.minimize(neg_log_lik, x0=np.array([0.0, 1.0]), method="BFGS")
    return -result.fun


def _csrsquared_from_cstatistic(
    cstatistic: float,
    prevalence: float,
    seed: int,
    n: int = 500_000,
) -> float:
    """Approximate the Cox-Snell R^2 implied by a reported C-statistic.

    Monte Carlo approximation from Riley, Van Calster & Collins (2020): assumes
    a binormal model for the linear predictor (events ~ N(mu, 1), non-events ~
    N(0, 1), with ``mu = sqrt(2) * qnorm(cstatistic)`` giving the target AUC),
    simulates a large sample at the target prevalence, fits a logistic
    regression of outcome on linear predictor, and reads off the fitted
    model's Cox-Snell R^2. `seed` makes the approximation reproducible.
    """
    rng = np.random.default_rng(seed)
    mu = float(np.sqrt(2) * stats.norm.ppf(cstatistic))

    n_events = round(prevalence * n)
    n_nonevents = n - n_events
    lp = np.concatenate(
        [
            rng.normal(loc=mu, scale=1.0, size=n_events),
            rng.normal(loc=0.0, scale=1.0, size=n_nonevents),
        ]
    )
    y = np.concatenate([np.ones(n_events), np.zeros(n_nonevents)])

    loglik_full = _fit_logistic_1d(lp, y)
    ybar = float(y.mean())
    loglik_null = n * (ybar * np.log(ybar) + (1 - ybar) * np.log(1 - ybar))

    return float(1 - np.exp((2 / n) * (loglik_null - loglik_full)))


def compute_pmsampsize_binary(
    parameters: int,
    prevalence: float,
    csrsquared: float | None = None,
    nagrsquared: float | None = None,
    cstatistic: float | None = None,
    shrinkage: float = 0.9,
    seed: int = 123456,
) -> BinarySampleSizeResult:
    """Minimum sample size for developing a binary outcome prediction model.

    What this solves
    -----------------
    Before collecting data to build a prediction model for a yes/no outcome
    (e.g. "will this patient be readmitted?"), you need to know how many
    subjects to enroll. Too few, and the model will overfit — look accurate
    on the data it was built on, then perform much worse on new patients.
    This calculates the minimum sample size needed to avoid that, based on
    how many candidate predictors you plan to consider and how well you
    expect the model to perform (which you provide via a rough estimate from
    a previous study — `csrsquared`, `nagrsquared`, or `cstatistic`, any one
    of which works; you don't need to understand the difference to use one).

    Implements the criteria of Riley et al. 2019 ("Minimum sample size
    required for developing a multivariable prediction model: Part II binary
    and time-to-event outcomes"):

    1. small overfitting, defined by shrinkage of predictor effects of 10% or
       less (i.e. `shrinkage` or higher);
    2. small absolute difference (<= 0.05) between the model's apparent and
       adjusted Nagelkerke's R^2;
    3. precise estimation (within +/- 0.05) of the average outcome risk in
       the population.

    Exactly one of `csrsquared`, `nagrsquared`, or `cstatistic` must be given,
    as the anticipated performance of the new model:

    - `csrsquared`: the expected Cox-Snell R^2 directly, e.g. taken from the
      adjusted R^2 of a previous model in the same field.
    - `nagrsquared`: the expected Nagelkerke's R^2 (Cox-Snell R^2 rescaled to
      [0, 1]), converted to Cox-Snell R^2 using `prevalence`.
    - `cstatistic`: a reported C-statistic (AUC), converted to an
      approximate Cox-Snell R^2 via the Monte Carlo method of Riley, Van
      Calster & Collins (2020) — see `_csrsquared_from_cstatistic`.

    Parameters
    ----------
    parameters : int
        Number of candidate predictor parameters for the new model.
    prevalence : float
        Overall outcome proportion expected in the development dataset, in
        (0, 1).
    csrsquared, nagrsquared, cstatistic : float, optional
        Exactly one must be given — see above.
    shrinkage : float, default=0.9
        Target shrinkage factor at internal validation, in (0, 1].
    seed : int, default=123456
        Seed for the C-statistic Monte Carlo approximation. Ignored unless
        `cstatistic` is given.

    Returns
    -------
    BinarySampleSizeResult
        `sample_size` (the maximum across all three criteria) plus the full
        per-criterion breakdown.

    Raises
    ------
    ValueError
        If:
        - not exactly one of csrsquared/nagrsquared/cstatistic is given
        - parameters < 1
        - prevalence is not in (0, 1)
        - shrinkage is not in (0, 1]
        - the resulting csrsquared is <= 0, >= 1, or exceeds the maximum
          Cox-Snell R^2 attainable at this prevalence
        - shrinkage is lower than csrsquared

    References
    ----------
    - Riley, R.D., Snell, K.I.E., Ensor, J., Burke, D.L., Harrell, F.E. Jr,
      Moons, K.G., & Collins, G.S. (2019). "Minimum sample size required for
      developing a multivariable prediction model: Part II binary and
      time-to-event outcomes." Statistics in Medicine, 38(7), 1276-1296.
    - Riley, R.D., Van Calster, B., & Collins, G.S. (2020). "A note on
      estimating the Cox-Snell R2 from a reported C statistic (AUROC) to
      inform sample size calculations for developing a prediction model
      with a binary outcome." Statistics in Medicine, 40(4), 859-864.

    """
    if parameters < 1:
        raise ValueError("parameters must be at least 1.")
    if not 0 < prevalence < 1:
        raise ValueError("prevalence must be between 0 and 1.")
    if not 0 < shrinkage <= 1:
        raise ValueError("shrinkage must be between 0 (exclusive) and 1.")

    given = [v is not None for v in (csrsquared, nagrsquared, cstatistic)]
    if sum(given) != 1:
        raise ValueError(
            "Exactly one of csrsquared, nagrsquared, or cstatistic must be given."
        )

    max_csrsquared = _max_csrsquared(prevalence)

    if csrsquared is not None:
        r2a = csrsquared
    elif nagrsquared is not None:
        if not 0 < nagrsquared < 1:
            raise ValueError("nagrsquared must be between 0 and 1.")
        r2a = nagrsquared * max_csrsquared
    else:
        if not 0.5 < cstatistic < 1:
            raise ValueError("cstatistic must be between 0.5 and 1.")
        r2a = _csrsquared_from_cstatistic(cstatistic, prevalence, seed)

    if not 0 < r2a < 1:
        raise ValueError(f"csrsquared must be between 0 and 1 (got {r2a}).")
    if shrinkage < r2a:
        raise ValueError(f"shrinkage ({shrinkage}) is lower than csrsquared ({r2a}).")
    if max_csrsquared < r2a:
        raise ValueError(
            f"csrsquared ({r2a}) is larger than the maximum Cox-Snell R^2 "
            f"attainable at this prevalence ({max_csrsquared:.3f})."
        )

    nag_r2 = r2a / max_csrsquared

    # Criterion 1 - shrinkage.
    n1 = int(np.ceil(parameters / ((shrinkage - 1) * np.log(1 - (r2a / shrinkage)))))
    shrinkage_1 = shrinkage
    events_1 = n1 * prevalence
    epp_1 = round(events_1 / parameters, 2)

    # Criterion 2 - small absolute difference in R^2 adjusted vs. apparent.
    s_4_small_diff = r2a / (r2a + (0.05 * max_csrsquared))
    n2 = int(
        np.ceil(
            parameters / ((s_4_small_diff - 1) * np.log(1 - (r2a / s_4_small_diff)))
        )
    )
    shrinkage_2 = s_4_small_diff
    events_2 = n2 * prevalence
    epp_2 = round(events_2 / parameters, 2)

    # Criterion 3 - precise estimation of the overall risk (intercept).
    n3 = int(np.ceil(((1.96 / 0.05) ** 2) * (prevalence * (1 - prevalence))))
    events_3 = n3 * prevalence
    epp_3 = round(events_3 / parameters, 2)

    shrinkage_3 = max(shrinkage, shrinkage_2)
    n_final = max(n1, n2, n3)
    shrinkage_final = shrinkage_3
    events_final = n_final * prevalence
    epp_final = round(events_final / parameters, 2)

    criteria = [
        BinaryCriterion(
            label="shrinkage",
            sample_size=n1,
            shrinkage=shrinkage_1,
            events=events_1,
            epp=epp_1,
        ),
        BinaryCriterion(
            label="small_r2_difference",
            sample_size=n2,
            shrinkage=shrinkage_2,
            events=events_2,
            epp=epp_2,
        ),
        BinaryCriterion(
            label="precise_intercept",
            sample_size=n3,
            shrinkage=shrinkage_3,
            events=events_3,
            epp=epp_3,
        ),
    ]

    return BinarySampleSizeResult(
        sample_size=n_final,
        shrinkage=shrinkage_final,
        parameters=parameters,
        csrsquared=r2a,
        max_csrsquared=max_csrsquared,
        nagrsquared=nag_r2,
        prevalence=prevalence,
        events=events_final,
        epp=epp_final,
        criteria=criteria,
    )
