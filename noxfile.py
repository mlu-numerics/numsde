"""Nox sessions mirroring the CI jobs, so every CI run can be reproduced locally.

Usage::

    nox -s tests        # fast suite on all installed Python versions
    nox -s slow         # statistical convergence tests
    nox -s gpu          # GPU tests (needs a CUDA device and PyTorch)
    nox -s lint docs
"""

import os

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "tests", "docs"]

PYTHONS = ["3.11", "3.12", "3.13", "3.14"]

# Fails the GPU session if PyTorch cannot use a CUDA device. Without this
# check the GPU tests would only be skipped and the session would pass.
CHECK_CUDA = """
import sys, torch
print(f"PyTorch {torch.__version__}, built for CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit(
        "No usable CUDA device. Check the driver with `nvidia-smi`: its CUDA "
        "version must be at least the one PyTorch was built for. Otherwise set "
        "NUMSDE_TORCH_INDEX to a matching wheel index, e.g. "
        "https://download.pytorch.org/whl/cu128, and run `nox -s gpu` again."
    )
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
"""


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


@nox.session(python=PYTHONS[-1], venv_backend="uv", reuse_venv=True)
def gpu(session: nox.Session) -> None:
    """Run GPU tests locally; requires a CUDA device.

    PyTorch wheels bundle the CUDA libraries, several GB on the first run, so
    the installation output is shown and the environment is reused across
    runs; NumSDE itself is reinstalled every time. Set ``NUMSDE_TORCH_INDEX``
    to a PyTorch wheel index that matches the local driver, for example
    ``https://download.pytorch.org/whl/cu128``; by default PyTorch comes from
    PyPI.
    """
    index = os.environ.get("NUMSDE_TORCH_INDEX")
    if index:
        session.install("torch", "--index-url", index, silent=False)
    session.install(
        ".[torch]", "--group", "test", "--reinstall-package", "numsde", silent=False
    )
    session.run("python", "-c", CHECK_CUDA)
    session.run("pytest", "-m", "gpu", "-rs", *session.posargs)


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
