from typing import Literal

from pydantic import BaseModel, model_validator

from .binary import BinaryCriterion, BinarySampleSizeResult, compute_binary_sample_size
from .continuous import (
    ContinuousCriterion,
    ContinuousSampleSizeResult,
    compute_continuous_sample_size,
)
from .survival import (
    SurvivalCriterion,
    SurvivalSampleSizeResult,
    compute_survival_sample_size,
)

__all__ = [
    "BinaryCriterion",
    "BinarySampleSizeResult",
    "ContinuousCriterion",
    "ContinuousSampleSizeResult",
    "PMSampleSize",
    "SurvivalCriterion",
    "SurvivalSampleSizeResult",
    "compute_binary_sample_size",
    "compute_continuous_sample_size",
    "compute_survival_sample_size",
]


class PMSampleSize(BaseModel):
    """Minimum sample size for developing a multivariable prediction model.

    What this solves
    -----------------
    Before collecting data to build any prediction model — continuous,
    binary, or time-to-event — you need to know how many subjects to
    enroll. Too few, and the model will overfit — look accurate on the data
    it was built on, then perform much worse in practice. This is a single
    entry point that dispatches to the right calculation for your outcome
    type. Prefer calling `compute_binary_sample_size`,
    `compute_continuous_sample_size`, or `compute_survival_sample_size`
    directly — this class exists for parity with the R `pmsampsize`
    package's single-function ergonomics.

    Implements the criteria proposed by Riley et al. 2019/2020 for minimum
    sample size when developing a new multivariable prediction model, for
    continuous, binary, or survival (time-to-event) outcomes.

    References
    ----------
    - Riley, R.D., Snell, K.I.E., Ensor, J., Burke, D.L., Harrell, F.E. Jr,
      Moons, K.G., & Collins, G.S. (2019). "Minimum sample size required for
      developing a multivariable prediction model: Part I continuous
      outcomes." Statistics in Medicine, 38(7), 1262-1275.
    - Riley, R.D., Snell, K.I.E., Ensor, J., Burke, D.L., Harrell, F.E. Jr,
      Moons, K.G., & Collins, G.S. (2019). "Minimum sample size required for
      developing a multivariable prediction model: Part II binary and
      time-to-event outcomes." Statistics in Medicine, 38(7), 1276-1296.
    - Riley, R.D., Van Calster, B., & Collins, G.S. (2020). "A note on
      estimating the Cox-Snell R2 from a reported C statistic (AUROC) to
      inform sample size calculations for developing a prediction model
      with a binary outcome." Statistics in Medicine, 40(4), 859-864.

    """

    outcome_type: Literal["continuous", "binary", "survival"]
    parameters: int
    shrinkage: float = 0.9

    rsquared: float | None = None
    intercept: float | None = None
    sd: float | None = None
    mmoe: float = 1.1

    prevalence: float | None = None
    csrsquared: float | None = None
    nagrsquared: float | None = None
    cstatistic: float | None = None
    seed: int = 123456

    rate: float | None = None
    timepoint: float | None = None
    meanfup: float | None = None

    @model_validator(mode="after")
    def _check_required_fields(self) -> "PMSampleSize":
        required: dict[str, tuple[str, ...]] = {
            "continuous": ("rsquared", "intercept", "sd"),
            "binary": ("prevalence",),
            "survival": ("rate", "timepoint", "meanfup"),
        }
        missing = [
            field
            for field in required[self.outcome_type]
            if getattr(self, field) is None
        ]
        if missing:
            raise ValueError(
                f"outcome_type={self.outcome_type!r} requires: {', '.join(missing)}"
            )
        return self

    def compute(
        self,
    ) -> BinarySampleSizeResult | ContinuousSampleSizeResult | SurvivalSampleSizeResult:
        if self.outcome_type == "continuous":
            return compute_continuous_sample_size(
                parameters=self.parameters,
                rsquared=self.rsquared,
                intercept=self.intercept,
                sd=self.sd,
                shrinkage=self.shrinkage,
                mmoe=self.mmoe,
            )
        if self.outcome_type == "binary":
            return compute_binary_sample_size(
                parameters=self.parameters,
                prevalence=self.prevalence,
                csrsquared=self.csrsquared,
                nagrsquared=self.nagrsquared,
                cstatistic=self.cstatistic,
                shrinkage=self.shrinkage,
                seed=self.seed,
            )
        return compute_survival_sample_size(
            parameters=self.parameters,
            rate=self.rate,
            timepoint=self.timepoint,
            meanfup=self.meanfup,
            csrsquared=self.csrsquared,
            nagrsquared=self.nagrsquared,
            shrinkage=self.shrinkage,
        )
