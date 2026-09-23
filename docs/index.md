# statsjunk

**A collection of statistical odds and ends.**

[![PyPI](https://img.shields.io/pypi/v/statsjunk)](https://pypi.org/project/statsjunk/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Framework-free Python functions for common statistical tasks — correlation, regression,
spatial autocorrelation, electoral fragmentation indices, and prediction-model sample size
calculations. Every function takes plain arrays and returns a typed
[Pydantic](https://docs.pydantic.dev/) result model; invalid input raises `ValueError` rather
than failing silently or returning `NaN`.

## Where to go next

- [Installation](installation.md) — one `pip install`, no optional extras
- [Quick Start](quickstart.md) — run a correlation and a sample-size calculation in a few lines
- [API Reference](reference/correlation.md) — every public function and class, with the paper it implements

## Modules

One folder per statistical domain, one module per technique inside it. Everything is also
re-exported from the top-level `statsjunk` package, so `from statsjunk import compute_pearson_correlation`
always works regardless of where a function actually lives.

| Module | Description |
|---|---|
| [`correlation`](reference/correlation.md) | Pearson and Spearman correlation with confidence intervals, plus the shared Fisher z-transformation |
| [`regression`](reference/regression.md) | Simple and multiple linear regression, with standardized coefficients and VIFs |
| [`spatial`](reference/spatial.md) | Global Moran's I spatial autocorrelation |
| [`elections`](reference/elections.md) | Laakso-Taagepera effective number of parties/candidates |
| [`samplesize`](reference/samplesize.md) | Minimum sample size for continuous, binary, and survival prediction models (a Python port of R's [`pmsampsize`](https://cran.r-project.org/package=pmsampsize)) |
