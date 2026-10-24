r"""NumSDE: numerical methods for stochastic differential equations.

NumSDE provides readable, tested reference implementations of numerical
schemes for Itô stochastic differential equations

.. math::

    \mathrm{d}X_t = f(t, X_t)\,\mathrm{d}t + g(t, X_t)\,\mathrm{d}W_t,

together with tools for empirical convergence studies. It is written for
university teaching and research.

The package is at an early stage of development; see the roadmap in the
developer documentation for the planned scope.
"""

from numsde._backend import RandomNumberGenerator, array_namespace, default_rng

try:
    from numsde._version import __version__
except ImportError:  # pragma: no cover - only without a build step
    __version__ = "0.0.0+unknown"

__all__ = [
    "RandomNumberGenerator",
    "__version__",
    "array_namespace",
    "default_rng",
]
