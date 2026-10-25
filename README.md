# NumSDE

[![CI](https://github.com/mlu-numerics/numsde/actions/workflows/ci.yml/badge.svg)](https://github.com/mlu-numerics/numsde/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/numsde/badge/?version=latest)](https://numsde.readthedocs.io)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

**Numerical methods for stochastic differential equations, for teaching and research.**

NumSDE provides readable, thoroughly tested reference implementations of numerical
schemes for Itô stochastic differential equations

$$
\mathrm{d}X_t = f(t, X_t)\,\mathrm{d}t + g(t, X_t)\,\mathrm{d}W_t,
\qquad X_t \in \mathbb{R}^d,\; W_t \in \mathbb{R}^m,
$$

together with tools for empirical strong and weak convergence studies and a
benchmark suite for comparing new schemes with established ones.

> **Status:** early development (milestone M0, foundations). The API is not
> usable yet. See the [architecture and roadmap](docs/development/architecture.md).

## Goals

- **Readable code:** each scheme's `step()` mirrors its textbook update rule.
- **Verified:** unit, property-based, statistical and regression tests; every
  scheme's empirical convergence order is checked against theory.
- **Didactic documentation:** a theory page for every scheme and executable
  tutorials.
- **Vectorised:** many paths are simulated at once; NumPy first, PyTorch (GPU)
  through the same code via the Python Array API standard.

## Installation

Once released:

```bash
pip install numsde              # core: NumPy and array-api-compat
pip install "numsde[symbolic]"  # + SymPy for symbolic derivatives
pip install "numsde[plot]"      # + matplotlib
pip install "numsde[torch]"     # + PyTorch backend
pip install "numsde[all]"
```

For development, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Citing NumSDE

If you use NumSDE in your research, please cite it using the metadata in
[CITATION.cff](CITATION.cff).

## License

NumSDE is distributed under the [BSD 3-Clause License](LICENSE).
