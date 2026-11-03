"""Benchmark problems for testing and comparing schemes.

The registry :data:`PROBLEMS` maps a short name to each problem class; every
problem has a theory page under :doc:`/theory/problems/index`. The benchmark
runner that compares schemes on these problems follows in milestone M4.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from numsde.benchmarks.problems import GeometricBrownianMotion, LinearSystem

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numsde.problems.base import ItoSDE

__all__ = ["PROBLEMS", "GeometricBrownianMotion", "LinearSystem", "get_problem"]

PROBLEMS: Final[Mapping[str, type[ItoSDE]]] = MappingProxyType(
    {
        "geometric_brownian_motion": GeometricBrownianMotion,
        "linear_system": LinearSystem,
    }
)
"""Registry of benchmark problems by name (read-only)."""


def get_problem(name: str, **params: Any) -> ItoSDE:
    """Create a registered benchmark problem.

    Parameters
    ----------
    name
        Key in :data:`PROBLEMS`.
    **params
        Parameters passed to the problem class.

    Returns
    -------
    numsde.ItoSDE
        The problem instance.

    Raises
    ------
    KeyError
        If `name` is not registered.

    Examples
    --------
    >>> get_problem("geometric_brownian_motion", sigma=0.5)
    GeometricBrownianMotion(mu=0.05, sigma=0.5, x0=1.0, t0=0.0)
    """
    try:
        cls = PROBLEMS[name]
    except KeyError:
        known = ", ".join(sorted(PROBLEMS))
        msg = f"Unknown benchmark problem {name!r}; registered problems: {known}."
        raise KeyError(msg) from None
    return cls(**params)
