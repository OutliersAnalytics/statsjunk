import numpy as np
from pydantic import BaseModel, Field
from scipy import stats


class ContinuousCriterion(BaseModel):
    """One row of the continuous sample size calculation's criterion breakdown."""

    label: str
    sample_size: int
    shrinkage: float
    spp: float


class ContinuousSampleSizeResult(BaseModel):
    """Minimum sample size for developing a continuous outcome prediction model."""

    sample_size: int
    shrinkage: float
    parameters: int
    rsquared: float = Field(ge=0, le=1)
    spp: float
    intercept: float
    intercept_low: float
    intercept_high: float
    intercept_mmoe: float
    criteria: list[ContinuousCriterion]


def _shrinkage_for_n(rsquared: float, parameters: int, n: int) -> float:
    """Expected shrinkage factor (van Houwelingen) for `n` observations."""
    return float(
        1
        + (
            (parameters - 2)
            / (
                n
                * np.log(
                    1 - (((rsquared * (n - parameters - 1)) + parameters) / (n - 1))
                )
            )
        )
    )


def compute_pmsampsize_continuous(
    parameters: int,
    rsquared: float,
    intercept: float,
    sd: float,
    shrinkage: float = 0.9,
    mmoe: float = 1.1,
) -> ContinuousSampleSizeResult:
    """Minimum sample size for developing a continuous outcome prediction model.

    What this solves
    -----------------
    Before collecting data to build a prediction model for a numeric outcome
    (e.g. predicting blood pressure from a set of risk factors), you need to
    know how many subjects to enroll. Too few, and the model will overfit —
    look accurate on the data it was built on, then perform much worse on
    new subjects. This calculates the minimum sample size needed to avoid
    that, based on how many candidate predictors you plan to consider and
    how well you expect the model to perform (a rough R^2 estimate from a
    previous study is enough).

    Implements the criteria of Riley et al. 2018 ("Minimum sample size
    required for developing a multivariable prediction model: Part I
    continuous outcomes"):

    1. small overfitting, defined by shrinkage of predictor effects of 10% or
       less (i.e. `shrinkage` or higher);
    2. small absolute difference (<= 0.05) between the model's apparent and
       adjusted R^2;
    3. precise estimation of the residual standard deviation (>= 234
       observations, per Riley et al.);
    4. precise estimation of the average outcome value (intercept), within a
       `mmoe` multiplicative margin of error.

    Parameters
    ----------
    parameters : int
        Number of candidate predictor parameters for the new model.
    rsquared : float
        Anticipated (adjusted) R^2 of the new model, in (0, 1).
    intercept : float
        Average outcome value in the population of interest. Must be
        nonzero, since the margin-of-error criterion is expressed as a ratio
        to it.
    sd : float
        Standard deviation of outcome values in the population.
    shrinkage : float, default=0.9
        Target shrinkage factor at internal validation, in (0, 1].
    mmoe : float, default=1.1
        Acceptable multiplicative margin of error for the intercept
        (1.1 = 10%). Must be greater than 1.

    Returns
    -------
    ContinuousSampleSizeResult
        `sample_size` (the maximum across all four criteria) plus the full
        per-criterion breakdown and the intercept's confidence interval at
        the final sample size.

    Raises
    ------
    ValueError
        If:
        - parameters < 1
        - rsquared is not in (0, 1)
        - sd is not positive
        - intercept is zero
        - shrinkage is not in (0, 1]
        - mmoe is not greater than 1

    References
    ----------
    - Riley, R.D., Snell, K.I.E., Ensor, J., Burke, D.L., Harrell, F.E. Jr,
      Moons, K.G., & Collins, G.S. (2019). "Minimum sample size required for
      developing a multivariable prediction model: Part I continuous
      outcomes." Statistics in Medicine, 38(7), 1262-1275.

    """
    if parameters < 1:
        raise ValueError("parameters must be at least 1.")
    if not 0 < rsquared < 1:
        raise ValueError("rsquared must be between 0 and 1.")
    if sd <= 0:
        raise ValueError("sd must be positive.")
    if intercept == 0:
        raise ValueError("intercept must be nonzero.")
    if not 0 < shrinkage <= 1:
        raise ValueError("shrinkage must be between 0 (exclusive) and 1.")
    if mmoe <= 1:
        raise ValueError("mmoe must be greater than 1.")

    # Criterion 1 - shrinkage.
    n1 = parameters + 2
    while _shrinkage_for_n(rsquared, parameters, n1) < shrinkage:
        n1 += 1
    shrinkage_1 = round(_shrinkage_for_n(rsquared, parameters, n1), 3)
    spp_1 = round(n1 / parameters, 2)

    # Criterion 2 - small absolute difference in R^2 adjusted vs. apparent.
    n2 = int(np.ceil(1 + ((parameters * (1 - rsquared)) / 0.05)))
    shrinkage_2 = round(_shrinkage_for_n(rsquared, parameters, n2), 3)
    spp_2 = round(n2 / parameters, 2)

    # Criterion 3 - precise estimation of the residual standard deviation.
    n3 = 234 + parameters
    shrinkage_3 = round(_shrinkage_for_n(rsquared, parameters, n3), 3)
    spp_3 = round(n3 / parameters, 2)

    # Criterion 4 - precise estimation of the intercept, within `mmoe`.
    n4 = max(n1, n2, n3)

    def _intercept_ci(n: int) -> tuple[float, float, float]:
        df = n - parameters - 1
        t_crit = float(stats.t.ppf(0.975, df))
        se = float(np.sqrt((sd**2 * (1 - rsquared)) / n))
        margin = t_crit * se
        low, high = intercept - margin, intercept + margin
        return low, high, high / intercept

    intercept_low, intercept_high, intercept_mmoe = _intercept_ci(n4)
    while intercept_mmoe > mmoe:
        n4 += 1
        intercept_low, intercept_high, intercept_mmoe = _intercept_ci(n4)
    shrinkage_4 = round(_shrinkage_for_n(rsquared, parameters, n4), 3)
    spp_4 = round(n4 / parameters, 2)

    n_final = max(n1, n2, n3, n4)
    shrinkage_final = round(_shrinkage_for_n(rsquared, parameters, n_final), 3)
    spp_final = round(n_final / parameters, 2)
    intercept_low, intercept_high, intercept_mmoe = _intercept_ci(n_final)

    criteria = [
        ContinuousCriterion(
            label="shrinkage", sample_size=n1, shrinkage=shrinkage_1, spp=spp_1
        ),
        ContinuousCriterion(
            label="small_r2_difference",
            sample_size=n2,
            shrinkage=shrinkage_2,
            spp=spp_2,
        ),
        ContinuousCriterion(
            label="precise_residual_sd",
            sample_size=n3,
            shrinkage=shrinkage_3,
            spp=spp_3,
        ),
        ContinuousCriterion(
            label="precise_intercept", sample_size=n4, shrinkage=shrinkage_4, spp=spp_4
        ),
    ]

    return ContinuousSampleSizeResult(
        sample_size=n_final,
        shrinkage=shrinkage_final,
        parameters=parameters,
        rsquared=rsquared,
        spp=spp_final,
        intercept=intercept,
        intercept_low=intercept_low,
        intercept_high=intercept_high,
        intercept_mmoe=intercept_mmoe,
        criteria=criteria,
    )
