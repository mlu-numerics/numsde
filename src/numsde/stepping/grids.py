"""Construction and validation of time grids.

Time grids are one-dimensional NumPy arrays of strictly increasing float64
times. They are small and live on the host, independent of the array backend
used for the simulated paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from numsde._backend import owned_read_only
from numsde._validation import finite_float, positive_int

__all__ = ["TimeGridLike", "as_time_grid", "uniform_grid"]

TimeGridLike: TypeAlias = Sequence[float] | npt.NDArray[np.floating]
"""Values accepted as a time grid: a sequence or 1-D array of floats."""


def uniform_grid(
    t_end: float, n_steps: int, t0: float = 0.0
) -> npt.NDArray[np.float64]:
    r"""Return the equidistant grid :math:`t_n = t_0 + n h`, :math:`h = (T - t_0)/N`.

    Parameters
    ----------
    t_end
        Final time :math:`T`, finite.
    n_steps
        Number of steps :math:`N`, a positive integer.
    t0
        Initial time :math:`t_0`, finite and smaller than `t_end`.

    Returns
    -------
    numpy.ndarray
        The ``n_steps + 1`` grid points; the end points are exact.

    Raises
    ------
    TypeError
        If `n_steps` is not an integer or a time is not a real number.
    ValueError
        If ``n_steps < 1``, a time is not finite, or ``t_end <= t0``.

    Examples
    --------
    >>> uniform_grid(1.0, 4)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    n_steps = positive_int(n_steps, "n_steps")
    t0 = finite_float(t0, "t0")
    t_end = finite_float(t_end, "t_end")
    if not t_end > t0:
        msg = f"t_end must be larger than t0, got t0={t0!r}, t_end={t_end!r}."
        raise ValueError(msg)
    with np.errstate(over="ignore", invalid="ignore"):
        grid = np.linspace(t0, t_end, n_steps + 1)
    _check_steps(grid)
    return grid


def as_time_grid(
    times: TimeGridLike, *, min_points: int = 2
) -> npt.NDArray[np.float64]:
    """Validate a time grid and return it as a read-only float64 array.

    Parameters
    ----------
    times
        Strictly increasing, finite times.
    min_points
        Minimal number of grid points.

    Returns
    -------
    numpy.ndarray
        A read-only copy of `times` with dtype float64.

    Raises
    ------
    ValueError
        If `times` is not one-dimensional, has fewer than `min_points`
        entries, contains non-finite values or is not strictly increasing.

    Examples
    --------
    >>> as_time_grid([0.0, 0.5, 1.0])
    array([0. , 0.5, 1. ])
    """
    grid = np.array(times, dtype=np.float64)
    if grid.ndim != 1:
        msg = f"A time grid must be one-dimensional, got shape {grid.shape}."
        raise ValueError(msg)
    if grid.size < min_points:
        msg = f"A time grid needs at least {min_points} points, got {grid.size}."
        raise ValueError(msg)
    if not np.all(np.isfinite(grid)):
        msg = "A time grid must contain only finite values."
        raise ValueError(msg)
    if np.any(grid[1:] <= grid[:-1]):
        msg = "A time grid must be strictly increasing."
        raise ValueError(msg)
    _check_steps(grid)
    return owned_read_only(grid, np, copy=False)  # type: ignore[no-any-return]


def _check_steps(grid: npt.NDArray[np.float64]) -> None:
    """Reject grids whose step sizes are not finite and strictly positive.

    Both failures come from finite grid points: the difference of two huge
    times overflows, and distinct points of a fine uniform grid far from zero
    round to the same float.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        steps = np.diff(grid)
    if not np.all(np.isfinite(steps)):
        msg = (
            "The step sizes of a time grid must be finite; the grid spans too "
            "large an interval."
        )
        raise ValueError(msg)
    if np.any(steps <= 0.0):
        msg = (
            "The step sizes of a time grid must be strictly positive; the steps "
            "are too small to be resolved in float64 relative to the grid points."
        )
        raise ValueError(msg)
