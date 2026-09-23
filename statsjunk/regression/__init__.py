from .multiple import (
    Coefficient,
    MultipleRegressionResult,
    compute_multiple_regression,
)
from .simple import RegressionResult, compute_linear_regression

__all__ = [
    "Coefficient",
    "MultipleRegressionResult",
    "RegressionResult",
    "compute_linear_regression",
    "compute_multiple_regression",
]
