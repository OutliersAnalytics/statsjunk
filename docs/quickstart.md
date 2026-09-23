# Quick Start

## Correlation

```python
from statsjunk import compute_pearson_correlation

result = compute_pearson_correlation([1, 2, 3, 4], [2, 4, 6, 8])
print(result.r, result.pvalue, result.low, result.high)
```

Only summary statistics on hand instead of the raw arrays? Use `compute_pearson_from_summary`:

```python
from statsjunk import compute_pearson_from_summary

result = compute_pearson_from_summary(r=0.62, n=48)
print(result.low, result.high)
```

## Sample size for a prediction model

```python
from statsjunk import PMSampleSize

result = PMSampleSize(
    outcome_type="binary",
    parameters=24,
    prevalence=0.174,
    cstatistic=0.89,
).compute()
print(result.sample_size)
```

See the [API Reference](reference/correlation.md) for every function and the paper it implements.
