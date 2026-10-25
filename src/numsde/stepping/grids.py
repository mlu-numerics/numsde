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
        Final time :math:`T`.
    n_steps
        Number of steps :math:`N`, at least 1.
    t0
        Initial time :math:`t_0`, smaller than `t_end`.

    Returns
    -------
    numpy.ndarray
        The ``n_steps + 1`` grid points; the end points are exact.

    Raises
    ------
    ValueError
        If ``n_steps < 1`` or ``t_end <= t0``.

    Examples
    --------
    >>> uniform_grid(1.0, 4)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    if int(n_steps) != n_steps or n_steps < 1:
        msg = f"n_steps must be a positive integer, got {n_steps!r}."
        raise ValueError(msg)
    if not t_end > t0:
        msg = f"t_end must be larger than t0, got t0={t0!r}, t_end={t_end!r}."
        raise ValueError(msg)
    return np.linspace(float(t0), float(t_end), int(n_steps) + 1)


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
    if np.any(np.diff(grid) <= 0.0):
        msg = "A time grid must be strictly increasing."
        raise ValueError(msg)
    grid.flags.writeable = False
    return grid
