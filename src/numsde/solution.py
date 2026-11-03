"""Container for the result of a simulation."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from numsde._backend import array_namespace, owned_read_only

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    import numpy.typing as npt

    from numsde.problems.base import SDEProblem
    from numsde.schemes.base import Scheme

__all__ = ["Solution"]


class Solution:
    r"""Approximations :math:`X_h^n` of a batch of paths at the saved times.

    Solutions are created by :meth:`Solver.solve <numsde.Solver.solve>`.

    Parameters
    ----------
    t
        Saved times (read-only NumPy array).
    x
        Values of shape ``(len(t), n_paths, dim)``; owned by the solution.
    problem
        The solved problem.
    scheme
        The scheme used.
    metadata
        Further information about the run.

    Examples
    --------
    >>> import numsde as ns
    >>> problem = ns.benchmarks.GeometricBrownianMotion()
    >>> solver = ns.Solver(problem, ns.schemes.EulerMaruyama())
    >>> solution = solver.solve(times=ns.uniform_grid(1.0, 8), n_paths=3, seed=0)
    >>> solution.t.shape, solution.x.shape
    ((9,), (9, 3, 1))
    >>> solution.metadata["scheme"]
    'Euler–Maruyama'
    """

    def __init__(
        self,
        t: npt.NDArray[np.float64],
        x: Any,
        *,
        problem: SDEProblem,
        scheme: Scheme,
        metadata: Mapping[str, Any],
    ) -> None:
        self._t = t
        self._x = owned_read_only(x, array_namespace(x), copy=False)
        self._problem = problem
        self._scheme = scheme
        self._metadata = MappingProxyType(dict(metadata))

    @property
    def t(self) -> npt.NDArray[np.float64]:
        """Saved times (read-only NumPy array)."""
        return self._t

    @property
    def x(self) -> Any:
        """Values :math:`X_h^n` at the saved times, ``(len(t), n_paths, dim)``.

        Immutable for NumPy; for other backends it must not be modified in
        place.
        """
        return self._x

    @property
    def final(self) -> Any:
        """Values at the last saved time, shape ``(n_paths, dim)``."""
        return self._x[-1, ...]

    @property
    def n_paths(self) -> int:
        """Number of paths."""
        return int(self._x.shape[1])

    @property
    def dim(self) -> int:
        """Dimension :math:`d` of the state."""
        return int(self._x.shape[2])

    @property
    def problem(self) -> SDEProblem:
        """The solved problem."""
        return self._problem

    @property
    def scheme(self) -> Scheme:
        """The scheme used."""
        return self._scheme

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Read-only information about the run.

        Keys: ``scheme``, ``n_steps``, ``n_paths``, ``backend``, ``dtype``,
        ``device``, ``save``, ``structure`` (set flags) and ``verified``
        (flags established by the library rather than declared).
        """
        return self._metadata

    def __repr__(self) -> str:
        """Return a short summary."""
        return (
            f"Solution(scheme={self._scheme.name!r}, n_times={self._t.size}, "
            f"n_paths={self.n_paths}, dim={self.dim})"
        )
