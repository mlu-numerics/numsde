"""Nox sessions mirroring the CI jobs, so every CI run can be reproduced locally.

Usage::

    nox -s tests        # fast suite on all installed Python versions
    nox -s slow         # statistical convergence tests
    nox -s gpu          # GPU tests (needs a CUDA device and PyTorch)
    nox -s lint docs
"""

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "tests", "docs"]

PYTHONS = ["3.11", "3.12", "3.13", "3.14"]


def install(session: nox.Session, *groups: str, extras: str = "") -> None:
    """Install the package (with extras) and the given dependency groups."""
    target = f".[{extras}]" if extras else "."
    group_args = [arg for group in groups for arg in ("--group", group)]
    session.install(target, *group_args)


@nox.session(python=PYTHONS)
def tests(session: nox.Session) -> None:
    """Run the fast test suite with coverage."""
    install(session, "test")
    session.run("pytest", "--cov", "--cov-report=term", *session.posargs)


@nox.session(python=PYTHONS[-1])
def tests_all_extras(session: nox.Session) -> None:
    """Run the fast suite with all optional dependencies (PyTorch on CPU)."""
    install(session, "test", extras="all")
    session.run("pytest", "--cov", "--cov-report=term", *session.posargs)


@nox.session(python="pypy3.11")
def pypy(session: nox.Session) -> None:
    """Run the fast suite on PyPy with the NumPy backend only."""
    install(session, "test")
    session.run("pytest", *session.posargs)


@nox.session(python=PYTHONS[-1])
def slow(session: nox.Session) -> None:
    """Run the statistical convergence tests."""
    install(session, "test")
    session.run("pytest", "-m", "slow", "-n", "auto", *session.posargs)


@nox.session(python=PYTHONS[-1], venv_backend="uv")
def gpu(session: nox.Session) -> None:
    """Run GPU tests locally; requires a CUDA device."""
    install(session, "test", extras="torch")
    session.run("pytest", "-m", "gpu", *session.posargs)


@nox.session(python=PYTHONS[-1])
def lint(session: nox.Session) -> None:
    """Run ruff, mypy and codespell."""
    install(session, "lint")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")
    session.run("mypy")
    session.run("codespell")


@nox.session(python=PYTHONS[-1])
def docs(session: nox.Session) -> None:
    """Build the documentation, treating warnings as errors."""
    install(session, "docs")
    session.run(
        "sphinx-build", "-W", "--keep-going", "-b", "html", "docs", "docs/_build/html"
    )


@nox.session(python=PYTHONS[-1])
def linkcheck(session: nox.Session) -> None:
    """Check external links in the documentation."""
    install(session, "docs")
    session.run("sphinx-build", "-b", "linkcheck", "docs", "docs/_build/linkcheck")
