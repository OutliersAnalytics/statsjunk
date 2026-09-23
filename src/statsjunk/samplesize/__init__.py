from typing import Literal

from pydantic import BaseModel, model_validator

from .binary import BinaryCriterion, BinarySampleSizeResult, compute_pmsampsize_binary
from .continuous import (
    ContinuousCriterion,
    ContinuousSampleSizeResult,
    compute_pmsampsize_continuous,
)
from .survival import (
    SurvivalCriterion,
    SurvivalSampleSizeResult,
    compute_pmsampsize_survival,
)

__all__ = [
    "BinaryCriterion",
    "BinarySampleSizeResult",
    "ContinuousCriterion",
    "ContinuousSampleSizeResult",
    "PMSampleSize",
    "SurvivalCriterion",
    "SurvivalSampleSizeResult",
    "compute_pmsampsize_binary",
    "compute_pmsampsize_continuous",
    "compute_pmsampsize_survival",
]


class PMSampleSize(BaseModel):
    """Minimum sample size for developing a multivariable prediction model.

    A single entry point over `compute_pmsampsize_binary`,
    `compute_pmsampsize_continuous`, and `compute_pmsampsize_survival`,
    dispatching on `outcome_type`. Prefer calling those functions directly —
    this class exists for parity with the R `pmsampsize` package's
    single-function ergonomics.

    Implements the criteria proposed by Riley et al. 2018/2019 for minimum
    sample size when developing a new multivariable prediction model, for
    continuous, binary, or survival (time-to-event) outcomes.

    References
    ----------
    - Riley RD, Snell KIE, Ensor J, Burke DL, Harrell FE Jr, Moons KG,
      Collins GS. Minimum sample size required for developing a
      multivariable prediction model: Part I continuous outcomes.
      Statistics in Medicine. 2019.
    - Riley RD, Snell KIE, Ensor J, Burke DL, Harrell FE Jr, Moons KG,
      Collins GS. Minimum sample size required for developing a
      multivariable prediction model: Part II binary and time-to-event
      outcomes. Statistics in Medicine. 2019.
    - Riley RD, Van Calster B, Collins GS. A note on estimating the Cox-Snell
      R^2 from a reported C statistic (AUROC) to inform sample size
      calculations for developing a prediction model with a binary outcome.
      Statistics in Medicine. 2020.

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
            return compute_pmsampsize_continuous(
                parameters=self.parameters,
                rsquared=self.rsquared,
                intercept=self.intercept,
                sd=self.sd,
                shrinkage=self.shrinkage,
                mmoe=self.mmoe,
            )
        if self.outcome_type == "binary":
            return compute_pmsampsize_binary(
                parameters=self.parameters,
                prevalence=self.prevalence,
                csrsquared=self.csrsquared,
                nagrsquared=self.nagrsquared,
                cstatistic=self.cstatistic,
                shrinkage=self.shrinkage,
                seed=self.seed,
            )
        return compute_pmsampsize_survival(
            parameters=self.parameters,
            rate=self.rate,
            timepoint=self.timepoint,
            meanfup=self.meanfup,
            csrsquared=self.csrsquared,
            nagrsquared=self.nagrsquared,
            shrinkage=self.shrinkage,
        )
