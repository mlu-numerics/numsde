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
- Sphinx documentation with MyST, API reference, errata page, bibliography and
  the architecture and roadmap document.
- Backend layer: `array_namespace` and `namespace_for` for Array API namespace
  resolution; `RandomNumberGenerator` and `default_rng` for reproducible,
  backend-independent random numbers.
