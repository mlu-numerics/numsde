# Contributing to NumSDE

Thank you for your interest in NumSDE. This guide describes how to set up a
development environment and what every contribution needs before it is merged.

## Development setup

NumSDE uses [uv](https://docs.astral.sh/uv/) for environments and
[nox](https://nox.thea.codes/) for reproducible test sessions.

```bash
git clone https://github.com/mlu-numerics/numsde.git
cd numsde
uv sync --group dev          # creates .venv with all development tools
uv run pre-commit install    # runs Ruff and mypy before every commit
```

Common tasks:

| Task | Command |
| --- | --- |
| Fast test suite | `uv run pytest` |
| Tests with coverage | `uv run pytest --cov` |
| Statistical (slow) tests | `uv run pytest -m slow` |
| GPU tests (local CUDA device) | `nox -s gpu` |
| Lint and type check | `nox -s lint` |
| Build the documentation | `nox -s docs` |
| Everything CI runs | `nox` |

Without network access, build the documentation with
`NUMSDE_DOCS_OFFLINE=1`, which skips links to external API documentation.

## Workflow

1. Open an issue or refer to a milestone in the
   [architecture and roadmap](docs/development/architecture.md).
2. Create a branch from `main`; never commit to `main` directly.
3. Open a pull request. It is merged once CI is green and it has been reviewed.

## Definition of done

A pull request is ready when:

- [ ] Tests cover the change: unit tests, property-based tests where an
      invariant exists, and for schemes a statistical convergence test
      (marker `slow`).
- [ ] Coverage stays at 100 % (lines and branches).
- [ ] `ruff check`, `ruff format --check` and `mypy` pass.
- [ ] Public functions and classes have NumPy-style docstrings with a runnable
      `Examples` section.
- [ ] New schemes have a theory page under `docs/theory/` with the update rule,
      convergence orders, assumptions and references.
- [ ] References are added to `docs/references.bib` with checked details.
- [ ] `CHANGELOG.md` has an entry under *Unreleased*.
- [ ] The documentation builds with `sphinx-build -W`.

## Corrections and errata

If a pull request fixes an error in results, formulas, convergence orders or
tests, add an entry to [docs/errata.md](docs/errata.md) and refer to its ID in
the changelog and the pull request. The errata page is the only place where a
correction is described.

## Style

- Project language is US English: code, comments, docstrings, documentation,
  commit messages and issues ("discretized", "vectorized", "behavior").
  `codespell` checks this in pre-commit and CI; add missing British-to-US
  pairs to `.codespell-dictionary.txt`.
- Mathematical names (`dW`, `h`, `X`) follow the notation of the theory pages.
  Stochastic processes are written $X(t)$, $W(t)$, never $X_t$: subscripts
  denote discretizations ($X_h$, $X_h^n$) and increments ($\Delta W_n$).
- Numerical code uses the Array API namespace (`xp = array_namespace(x)`),
  never `numpy` directly, so that every backend runs the same code.
  Backend-specific code lives in `numsde._backend` only.
