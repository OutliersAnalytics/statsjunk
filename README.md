# statsjunk

A collection of statistical odds and ends.

Framework-free Python functions for common statistical tasks — correlation, regression,
spatial autocorrelation, electoral fragmentation indices, and (soon) sample size
calculations. Every function takes plain arrays and returns a typed [Pydantic](https://docs.pydantic.dev/)
result model; invalid input raises `ValueError` rather than failing silently.

## Installing

```sh
pip install statsjunk
```

## Usage

```python
from statsjunk import compute_pearson_correlation

result = compute_pearson_correlation([1, 2, 3, 4], [2, 4, 6, 8])
print(result.r, result.pvalue)
```

## What's here

One folder per statistical domain, one module per technique inside it. Everything is also
re-exported from the top-level `statsjunk` package, so `from statsjunk import compute_pearson_correlation`
always works regardless of where a function actually lives.

- `correlation` — `pearson`: Pearson correlation, with confidence intervals, and a
  summary-statistics variant (`r`, `n`) for when the raw arrays aren't available. `fisher`: the
  Fisher z-transformation used internally by `pearson`, also usable standalone.
- `regression` — `simple`: simple linear regression with slope diagnostics. `multiple`: ordinary
  least squares multiple linear regression, with standardized coefficients and variance inflation
  factors.
- `spatial` — `morans_i`: global Moran's I spatial autocorrelation.
- `elections` — `laakso_taagepera`: effective number of parties/candidates.
- `samplesize` — a Python port of [`pmsampsize`](https://cran.r-project.org/package=pmsampsize)
  (`binary`, `continuous`, `survival`), for minimum sample size calculations when developing a
  prediction model (Riley et al. 2019, 2020).

## Developing

```sh
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format .
```
