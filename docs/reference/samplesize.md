# statsjunk.samplesize

`PMSampleSize` is a single entry point that dispatches to the right calculation based on
`outcome_type`. Prefer calling `compute_binary_sample_size`, `compute_continuous_sample_size`,
or `compute_survival_sample_size` directly — the class exists for parity with the R
`pmsampsize` package's single-function ergonomics.

::: statsjunk.samplesize.PMSampleSize

::: statsjunk.samplesize.binary
    options:
      members:
        - compute_binary_sample_size
        - BinarySampleSizeResult
        - BinaryCriterion

::: statsjunk.samplesize.continuous
    options:
      members:
        - compute_continuous_sample_size
        - ContinuousSampleSizeResult
        - ContinuousCriterion

::: statsjunk.samplesize.survival
    options:
      members:
        - compute_survival_sample_size
        - SurvivalSampleSizeResult
        - SurvivalCriterion
