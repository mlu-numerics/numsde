r"""NumSDE: numerical methods for stochastic differential equations.

NumSDE provides readable, tested reference implementations of numerical
schemes for Itô stochastic differential equations

.. math::

    \mathrm{d}X(t) = f(t, X(t))\,\mathrm{d}t + g(t, X(t))\,\mathrm{d}W(t),

together with tools for empirical convergence studies. It is written for
university teaching and research.

The package is at an early stage of development; see the roadmap in the
developer documentation for the planned scope.
"""

from numsde._backend import RandomNumberGenerator, array_namespace, default_rng
from numsde.noise import BrownianPath, NoiseProcess, WienerProcess
from numsde.stepping import uniform_grid

try:
    from numsde._version import __version__
except ImportError:  # pragma: no cover - only without a build step
    __version__ = "0.0.0+unknown"

__all__ = [
    "BrownianPath",
    "NoiseProcess",
    "RandomNumberGenerator",
    "WienerProcess",
    "__version__",
    "array_namespace",
    "default_rng",
    "uniform_grid",
]
