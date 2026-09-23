# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`statsjunk` is a public, general-purpose Python package of statistical functions, published on PyPI.
It has no connection to any specific application — it must never depend on or reference any
Outliers Analytics product, data model, or domain concept. Code identifiers, docstrings, error
messages, README, and commit messages are all in English, since this is a public package with no
assumed audience.

## Scope

Only general-purpose statistics belong here — functions any Python project could plausibly need
(correlation, regression, spatial statistics, sample size calculations, and so on). If a function
is specific to one product or dataset, it does not belong in `statsjunk`, no matter where that
product's own code happens to live.

## Commands

```sh
uv sync --dev          # install deps
uv run pytest           # run all tests
uv run pytest tests/test_correlation.py::test_name   # a single test
uv run ruff check .
uv run ruff format .
```

## Architecture

`src/statsjunk/` is flat — the whole package already is "analytics", so there's no `analytics/`
subpackage. Each module holds one family of related functions that take plain arrays/sequences and
return a typed Pydantic result model. Functions raise `ValueError` for invalid input (mismatched
lengths, too few observations, constant series, out-of-range parameters) rather than returning
error codes or `None` — callers are expected to let these propagate or translate them at their own
boundary.

Keep this package free of any web framework, database, or I/O dependency — it should be importable
and testable with nothing but `numpy`/`scipy`/`pydantic`.

## Releasing

Pushing a `v*` tag runs the test suite and publishes to PyPI via trusted publishing (OIDC) —
see `.github/workflows/publish.yml`. There is no manually-stored PyPI token.
