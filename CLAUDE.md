# Notes for AI assistants working on NumSDE

These conventions apply to every change in this repository.

## Sources of truth

- Scope, architecture, roadmap and decisions: `docs/development/architecture.md`.
  Record new decisions in its decision log as part of the same pull request.
- Corrections of errors: `docs/errata.md` only. Changelog, pull requests and
  theory pages refer to the errata ID and never describe the correction again.
- Contribution rules and definition of done: `CONTRIBUTING.md`.

## Conventions

- Project language is US English (code, docs, commits, issues), even when the
  conversation with the maintainer is in German: "discretized", "vectorized",
  "behavior", "neighbor". This matches the companion book project; codespell
  enforces it, and `.codespell-dictionary.txt` holds the numerical-analysis
  terms its built-in dictionary lacks.
- Never commit to `main`; work on a branch and open a pull request.
- Numerical code uses the Array API namespace (`xp = array_namespace(...)`);
  backend-specific code lives only in `src/numsde/_backend/`.
- Every public object has a NumPy-style docstring with a runnable example
  (doctests run in the test suite).
- Coverage target is 100 % lines and branches; tests for slow statistical
  checks carry `@pytest.mark.slow`, GPU tests `@pytest.mark.gpu`.
- `main` requires the single status check `CI passed`. A new job in
  `.github/workflows/ci.yml` must be added to the `needs` list of the
  `ci-passed` job, otherwise it does not gate merging.
- Only add bibliography entries whose details were checked against the
  publisher's page or the article.
- Notation: stochastic processes are written `X(t)`, `W(t)`, never `X_t`;
  subscripts denote discretizations (`X_h`, `X_h^n`) and increments
  (`ΔW_n`); components of vector processes use superscripts (`W^j(t)`).
- The default strong error is the error uniformly on the grid,
  `(E[max_n |X_h^n - X(t_n)|^p])^(1/p)`.

## Checks before opening a pull request

```bash
uv run pytest --cov
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run codespell
NUMSDE_DOCS_OFFLINE=1 uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```
