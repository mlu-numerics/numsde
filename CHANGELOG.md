# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Corrections of errors are described in [docs/errata.md](docs/errata.md);
entries here only refer to their errata ID.

## [Unreleased]

### Added

- Project infrastructure: packaging with hatchling and hatch-vcs, uv, nox,
  pre-commit, Ruff, mypy (strict), pytest with Hypothesis and coverage.
- GitHub Actions workflows for tests and documentation.
- Spell checking with codespell, enforcing US English as in the companion book.
- Sphinx documentation with MyST, API reference, errata page, bibliography and
  the architecture and roadmap document.
- Theory section for benchmark problems: every problem of the benchmark suite
  gets its own theory page, with a fixed structure.
- `WienerProcess` and `BrownianPath`: sampling of independent Brownian paths
  on arbitrary grids, refinement with the Brownian bridge, insertion of
  arbitrary times, and coarsening that sums increments exactly.
- `uniform_grid` and time-grid validation in `numsde.stepping`; grids whose
  step sizes overflow or cannot be resolved in float64 are rejected.
- `BrownianPath` contract: values are owned, immutable for NumPy (they cannot
  be made writeable again) and never shared between paths;
  `insert` and `refine` are deterministic functions of the path's seed and
  the requested times; values must be real floating-point with at least one
  path and one dimension.
- Consistent input validation: `TypeError` for wrong types (including `bool`
  and `float` where an integer is expected), `ValueError` for invalid values.
- CI job that builds the wheel, checks its metadata and runs the test suite
  against the installed package.
- Theory page on Brownian motion and the Brownian bridge.
- Backend layer: `array_namespace` and `namespace_for` for Array API namespace
  resolution; `RandomNumberGenerator` and `default_rng` for reproducible,
  backend-independent random numbers.
- Problems: `ItoSDE` (base class `SDEProblem`) for user-defined initial value
  problems, `NoiseStructure` with declared and verified flags, `Calculus`.
- Step quantities: `StepNoise` and the `NoiseSource` interface;
  `BrownianPath` is a noise source.
- `EulerMaruyama` scheme with base classes `Scheme` and `OneStepScheme`.
- `Solver` and `Solution`: vectorized time loop over a batch of paths, checks
  of scheme, noise and coefficient shapes before the loop, `save="final"`.
- Benchmark problems `GeometricBrownianMotion` and `LinearSystem` with exact
  solutions, and the registry `numsde.benchmarks.PROBLEMS`.
- `asarray_like` to convert problem parameters to the backend of the state.
- Theory pages for the Euler–Maruyama scheme, geometric Brownian motion and
  the linear system.
