r"""Wiener process and sampled Brownian paths.

A :class:`BrownianPath` stores the values :math:`W(t_0), \dots, W(t_N)` of
`n_paths` independent :math:`m`-dimensional Brownian motions on a common time
grid. New grid points are filled in with the Brownian bridge, conditioned on
the values already sampled, so a path can be refined at any time without
changing what is already known about it. Coarsening keeps every `factor`-th
grid point; the increments of the coarse path are then exactly the sums of the
fine increments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import array_api_compat
import numpy as np

from numsde._backend import (
    RandomNumberGenerator,
    array_namespace,
    default_rng,
    namespace_for,
)
from numsde.noise.base import NoiseProcess
from numsde.stepping.grids import as_time_grid

if TYPE_CHECKING:
    from types import ModuleType

    import numpy.typing as npt

    from numsde._backend import SeedLike
    from numsde.stepping.grids import TimeGridLike

__all__ = ["BrownianPath", "WienerProcess"]


class BrownianPath:
    r"""Sampled values of independent Brownian motions on a common time grid.

    Most users obtain a path from :meth:`WienerProcess.sample_path`. The
    constructor is useful to wrap values sampled elsewhere, for example
    reference data from an existing script.

    Parameters
    ----------
    times
        Strictly increasing grid :math:`t_0 < \dots < t_N`; a single point is
        allowed.
    values
        Array of shape ``(N + 1, n_paths, m)`` with the values
        :math:`W(t_n)` of each path, in any Array API namespace.
    rng
        Generator used for later refinements, or a seed for a new one.

    Raises
    ------
    ValueError
        If `values` does not have shape ``(len(times), n_paths, m)``.

    See Also
    --------
    WienerProcess.sample_path : Sample a new path.

    Examples
    --------
    >>> import numpy as np
    >>> path = BrownianPath([0.0, 1.0], np.array([[[0.0]], [[0.3]]]))
    >>> path.increments[:, 0, 0]
    array([0.3])
    """

    def __init__(
        self,
        times: TimeGridLike,
        values: Any,
        *,
        rng: SeedLike | RandomNumberGenerator = None,
    ) -> None:
        self._times = as_time_grid(times, min_points=1)
        self._xp = array_namespace(values)
        if values.ndim != 3 or values.shape[0] != self._times.size:
            msg = (
                "values must have shape (len(times), n_paths, dim) = "
                f"({self._times.size}, n_paths, dim), got {tuple(values.shape)}."
            )
            raise ValueError(msg)
        self._values = values
        self._rng = default_rng(rng)

    # ------------------------------------------------------------------ data

    @property
    def times(self) -> npt.NDArray[np.float64]:
        r"""Grid points :math:`t_0, \dots, t_N` (read-only NumPy array)."""
        return self._times

    @property
    def values(self) -> Any:
        """Values :math:`W(t_n)`, shape ``(n_steps + 1, n_paths, dim)``."""
        return self._values

    @property
    def increments(self) -> Any:
        r"""Increments :math:`\Delta W_n = W(t_{n+1}) - W(t_n)`.

        Shape ``(n_steps, n_paths, dim)``.
        """
        return self._values[1:, ...] - self._values[:-1, ...]

    @property
    def step_sizes(self) -> npt.NDArray[np.float64]:
        r"""Step sizes :math:`h_n = t_{n+1} - t_n` (NumPy array)."""
        return np.diff(self._times)

    @property
    def n_steps(self) -> int:
        """Number of steps :math:`N` of the grid."""
        return int(self._times.size) - 1

    @property
    def n_paths(self) -> int:
        """Number of independent paths."""
        return int(self._values.shape[1])

    @property
    def dim(self) -> int:
        """Dimension :math:`m` of the Brownian motion."""
        return int(self._values.shape[2])

    @property
    def t0(self) -> float:
        """First grid point."""
        return float(self._times[0])

    @property
    def t_end(self) -> float:
        """Last grid point."""
        return float(self._times[-1])

    @property
    def rng(self) -> RandomNumberGenerator:
        """Generator used for refinements."""
        return self._rng

    # ------------------------------------------------------- grid operations

    def coarsen(self, factor: int) -> BrownianPath:
        """Keep every `factor`-th grid point.

        The increments of the coarse path are the sums of `factor`
        consecutive increments of this path, so both paths describe the same
        Brownian motion.

        Parameters
        ----------
        factor
            Positive integer that divides :attr:`n_steps`.

        Returns
        -------
        BrownianPath
            The coarse path. It shares the random number generator with this
            path.

        Raises
        ------
        ValueError
            If `factor` is not a positive integer dividing :attr:`n_steps`.

        Examples
        --------
        >>> fine = WienerProcess().sample_path([0.0, 0.25, 0.5, 0.75, 1.0], seed=1)
        >>> fine.coarsen(2).times
        array([0. , 0.5, 1. ])
        """
        _check_factor(factor)
        if self.n_steps % factor != 0:
            msg = f"factor {factor} does not divide the number of steps {self.n_steps}."
            raise ValueError(msg)
        return BrownianPath(
            self._times[::factor], self._values[::factor, ...], rng=self._rng
        )

    def refine(
        self, factor: int, *, rng: SeedLike | RandomNumberGenerator = None
    ) -> BrownianPath:
        """Split every step into `factor` equal substeps.

        The new values are sampled from the Brownian bridge between the
        existing ones, so ``path.refine(k).coarsen(k)`` reproduces `path`
        exactly.

        Parameters
        ----------
        factor
            Positive integer.
        rng
            Generator or seed for the new samples; defaults to the path's own
            generator.

        Returns
        -------
        BrownianPath
            The refined path.

        Examples
        --------
        >>> path = WienerProcess().sample_path([0.0, 1.0], seed=3)
        >>> path.refine(4).n_steps
        4
        """
        _check_factor(factor)
        fractions = np.arange(1, factor) / factor
        left = self._times[:-1, None]
        new_times = (left + fractions[None, :] * np.diff(self._times)[:, None]).ravel()
        return self.insert(new_times, rng=rng)

    def insert(
        self,
        times: TimeGridLike,
        *,
        rng: SeedLike | RandomNumberGenerator = None,
    ) -> BrownianPath:
        r"""Add grid points, keeping all values already sampled.

        Points inside the grid are sampled from the Brownian bridge,
        conditioned on the neighboring values; points after :attr:`t_end`
        extend the path with independent increments. Points already on the
        grid are ignored.

        Parameters
        ----------
        times
            New grid points, not smaller than :attr:`t0`; any order.
        rng
            Generator or seed for the new samples; defaults to the path's own
            generator.

        Returns
        -------
        BrownianPath
            The path on the merged grid. It keeps this path's generator.

        Raises
        ------
        ValueError
            If a time is not finite or lies before :attr:`t0`.

        Examples
        --------
        >>> path = WienerProcess().sample_path([0.0, 1.0], seed=0)
        >>> longer = path.insert([0.3, 2.0])
        >>> longer.times
        array([0. , 0.3, 1. , 2. ])
        >>> bool((longer.values[[0, 2]] == path.values).all())
        True
        """
        new = np.unique(np.asarray(times, dtype=np.float64).ravel())
        if not np.all(np.isfinite(new)):
            msg = "Inserted times must be finite."
            raise ValueError(msg)
        if new.size and new[0] < self.t0:
            msg = f"Cannot insert times before t0={self.t0}, got {new[0]}."
            raise ValueError(msg)
        new = new[~np.isin(new, self._times)]
        if new.size == 0:
            return self
        generator = self._rng if rng is None else default_rng(rng)

        inside = new[new < self.t_end]
        beyond = new[new > self.t_end]
        blocks = [self._values]
        block_times = [self._times]
        if inside.size:
            blocks.append(self._bridge(inside, generator))
            block_times.append(inside)
        if beyond.size:
            blocks.append(self._extend(beyond, generator))
            block_times.append(beyond)

        xp = self._xp
        merged_times = np.concatenate(block_times)
        order = np.argsort(merged_times, kind="stable")
        index = xp.asarray(order, device=array_api_compat.device(self._values))
        merged = xp.take(xp.concat(blocks, axis=0), index, axis=0)
        return BrownianPath(merged_times[order], merged, rng=self._rng)

    # ------------------------------------------------------------- sampling

    def _column(self, coefficients: npt.NDArray[np.float64]) -> Any:
        """Convert host coefficients to an array of shape ``(k, 1, 1)``."""
        return self._xp.asarray(
            coefficients[:, None, None],
            dtype=self._values.dtype,
            device=array_api_compat.device(self._values),
        )

    def _normal(self, k: int, rng: RandomNumberGenerator) -> Any:
        """Draw standard normal samples of shape ``(k, n_paths, dim)``."""
        return rng.standard_normal(
            (k, self.n_paths, self.dim),
            xp=self._xp,
            dtype=self._values.dtype,
            device=array_api_compat.device(self._values),
        )

    def _rows(self, values: Any, rows: npt.NDArray[np.intp]) -> Any:
        """Select rows of `values` given by host indices."""
        index = self._xp.asarray(rows, device=array_api_compat.device(values))
        return self._xp.take(values, index, axis=0)

    def _bridge(self, new: npt.NDArray[np.float64], rng: RandomNumberGenerator) -> Any:
        r"""Sample :math:`W` at sorted times strictly inside the grid.

        Several new points in one interval are sampled from left to right,
        each conditioned on its left neighbor (old or new) and the right
        grid point. By the Markov property this gives the exact joint law.
        """
        xp = self._xp
        old = self._times
        right = np.searchsorted(old, new, side="right")
        k = new.size
        first_in_interval = np.r_[True, right[1:] != right[:-1]]
        group_start = np.maximum.accumulate(
            np.where(first_in_interval, np.arange(k), 0)
        )
        rank = np.arange(k) - group_start
        noise = self._normal(k, rng)

        results: list[Any] = []
        positions: list[npt.NDArray[np.intp]] = []
        for r in range(int(np.max(rank)) + 1):
            idx = np.flatnonzero(rank == r)
            t = new[idx]
            t_right = old[right[idx]]
            w_right = self._rows(self._values, right[idx])
            if r == 0:
                t_left = old[right[idx] - 1]
                w_left = self._rows(self._values, right[idx] - 1)
            else:
                t_left = new[idx - 1]
                w_left = self._rows(
                    results[-1], np.searchsorted(positions[-1], idx - 1)
                )
            weight = (t - t_left) / (t_right - t_left)
            std = np.sqrt((t - t_left) * (t_right - t) / (t_right - t_left))
            values = (
                w_left
                + self._column(weight) * (w_right - w_left)
                + self._column(std) * self._rows(noise, idx)
            )
            results.append(values)
            positions.append(idx)

        sampled = xp.concat(results, axis=0)
        return self._rows(sampled, np.argsort(np.concatenate(positions)))

    def _extend(self, new: npt.NDArray[np.float64], rng: RandomNumberGenerator) -> Any:
        """Sample :math:`W` at sorted times after the last grid point."""
        steps = np.diff(np.r_[self.t_end, new])
        increments = self._column(np.sqrt(steps)) * self._normal(new.size, rng)
        return self._values[-1:, ...] + self._xp.cumulative_sum(increments, axis=0)

    def __repr__(self) -> str:
        """Return a short summary of the grid and the sample size."""
        return (
            f"BrownianPath(t0={self.t0}, t_end={self.t_end}, n_steps={self.n_steps}, "
            f"n_paths={self.n_paths}, dim={self.dim})"
        )


class WienerProcess(NoiseProcess):
    r"""Standard :math:`m`-dimensional Wiener process (Brownian motion).

    Parameters
    ----------
    dim
        Dimension :math:`m` of the process, at least 1.

    Examples
    --------
    >>> W = WienerProcess(dim=2)
    >>> path = W.sample_path([0.0, 0.5, 1.0], n_paths=1000, seed=42)
    >>> path.increments.shape
    (2, 1000, 2)
    """

    def sample_path(
        self,
        times: TimeGridLike,
        n_paths: int = 1,
        *,
        seed: SeedLike | RandomNumberGenerator = None,
        backend: str | ModuleType | None = None,
        dtype: Any = None,
        device: Any = None,
    ) -> BrownianPath:
        r"""Sample independent Brownian paths on a time grid.

        The paths start at :math:`W(t_0) = 0` and have independent increments
        :math:`\Delta W_n \sim \mathcal{N}(0, h_n I_m)`.

        Parameters
        ----------
        times
            Strictly increasing grid with at least two points.
        n_paths
            Number of independent paths, at least 1.
        seed
            Seed or generator; the same seed gives the same path on every
            backend.
        backend
            Array namespace or backend name such as ``"numpy"`` or
            ``"torch"``. Defaults to NumPy.
        dtype
            Floating-point dtype of the backend; defaults to double precision.
        device
            Device of the returned arrays, e.g. ``"cuda"``.

        Returns
        -------
        BrownianPath
            The sampled paths, values of shape ``(len(times), n_paths, dim)``.

        Raises
        ------
        ValueError
            If the grid is invalid or `n_paths` is not a positive integer.

        Examples
        --------
        >>> from numsde.stepping import uniform_grid
        >>> path = WienerProcess().sample_path(uniform_grid(1.0, 8), n_paths=3, seed=7)
        >>> path.values.shape
        (9, 3, 1)
        """
        grid = as_time_grid(times)
        if isinstance(n_paths, bool) or int(n_paths) != n_paths or n_paths < 1:
            msg = f"n_paths must be a positive integer, got {n_paths!r}."
            raise ValueError(msg)
        xp = namespace_for(backend) if isinstance(backend, str) else backend
        if xp is None:
            xp = namespace_for("numpy")
        start = xp.zeros(
            (1, int(n_paths), self.dim),
            dtype=xp.float64 if dtype is None else dtype,
            device=device,
        )
        return BrownianPath(grid[:1], start, rng=seed).insert(grid[1:])


def _check_factor(factor: int) -> None:
    """Raise unless `factor` is a positive integer."""
    if isinstance(factor, bool) or int(factor) != factor or factor < 1:
        msg = f"factor must be a positive integer, got {factor!r}."
        raise ValueError(msg)
