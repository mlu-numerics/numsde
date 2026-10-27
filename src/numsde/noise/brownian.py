r"""Wiener process and sampled Brownian paths.

A :class:`BrownianPath` is one fixed realization of `n_paths` independent
:math:`m`-dimensional Brownian motions on a common time grid. Its contract:

* **Values are owned and immutable.** The constructor copies its input, and
  every operation returns a new path that shares no memory with the old one.
  NumPy values live in an immutable buffer: neither the arrays returned by
  :attr:`~BrownianPath.values` and :attr:`~BrownianPath.times` nor views of
  them can be made writeable. Other backends (PyTorch) have no read-only
  arrays; there the contract only rules out aliasing, and values must not be
  modified in place.
* **Operations are pure functions.** A path carries an immutable seed
  sequence instead of a random number generator. :meth:`BrownianPath.insert`
  and :meth:`BrownianPath.refine` derive their random numbers from that seed
  and the requested times, so the same call always returns the same path.
* **Refinement keeps what is known.** New grid points are sampled from the
  Brownian bridge, conditioned on the values already on the grid; existing
  values never change. Coarsening keeps every `factor`-th grid point, so the
  coarse increments are exactly the sums of the fine increments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import array_api_compat
import numpy as np

from numsde._backend import (
    RandomNumberGenerator,
    array_namespace,
    derive_seed_sequence,
    device_of,
    frozen_seed_sequence,
    is_real_floating,
    namespace_for,
    owned_read_only,
)
from numsde._validation import positive_int
from numsde.noise.base import NoiseProcess
from numsde.noise.steps import NoiseSource, StepNoise
from numsde.stepping.grids import as_time_grid

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

    import numpy.typing as npt

    from numsde._backend import SeedLike
    from numsde.stepping.grids import TimeGridLike

__all__ = ["BrownianPath", "WienerProcess"]


class BrownianPath(NoiseSource):
    r"""One realization of independent Brownian motions on a common time grid.

    Most users obtain a path from :meth:`WienerProcess.sample_path`. The
    constructor is useful to wrap values sampled elsewhere, for example
    reference data from an existing script. A path is a
    :class:`~numsde.NoiseSource`: :meth:`~numsde.NoiseSource.steps` yields
    the increments of consecutive steps, so a path can drive a solver.

    Parameters
    ----------
    times
        Strictly increasing grid :math:`t_0 < \dots < t_N`; a single point is
        allowed.
    values
        Real floating-point array of shape ``(N + 1, n_paths, m)`` with the
        values :math:`W(t_n)`, in any Array API namespace. It is copied.
    seed
        Seed from which later refinements derive their random numbers.
        ``None`` draws fresh entropy, so refinements are reproducible for
        this path object but not across program runs.

    Raises
    ------
    TypeError
        If `values` does not have a real floating-point dtype.
    ValueError
        If `values` does not have shape ``(len(times), n_paths, m)`` with
        ``n_paths >= 1`` and ``m >= 1``.

    See Also
    --------
    WienerProcess.sample_path : Sample a new path.

    Examples
    --------
    >>> import numpy as np
    >>> path = BrownianPath([0.0, 1.0], np.array([[[0.0]], [[0.3]]]), seed=1)
    >>> path.increments[:, 0, 0]
    array([0.3])
    >>> path.values.flags.writeable
    False
    """

    def __init__(
        self,
        times: TimeGridLike,
        values: Any,
        *,
        seed: SeedLike = None,
    ) -> None:
        xp = array_namespace(values)
        self._init(
            as_time_grid(times, min_points=1),
            owned_read_only(values, xp, copy=True),
            frozen_seed_sequence(seed),
            xp,
        )

    @classmethod
    def _from_owned(
        cls,
        times: npt.NDArray[np.float64],
        values: Any,
        seed: np.random.SeedSequence,
        xp: ModuleType,
    ) -> BrownianPath:
        """Wrap arrays created internally and referenced nowhere else."""
        path = cls.__new__(cls)
        path._init(times, owned_read_only(values, xp, copy=False), seed, xp)
        return path

    def _init(
        self,
        times: npt.NDArray[np.float64],
        values: Any,
        seed: np.random.SeedSequence,
        xp: ModuleType,
    ) -> None:
        if not is_real_floating(values, xp):
            msg = f"values must have a real floating-point dtype, got {values.dtype}."
            raise TypeError(msg)
        if (
            values.ndim != 3
            or values.shape[0] != times.size
            or values.shape[1] < 1
            or values.shape[2] < 1
        ):
            msg = (
                "values must have shape (len(times), n_paths, dim) = "
                f"({times.size}, n_paths >= 1, dim >= 1), got {tuple(values.shape)}."
            )
            raise ValueError(msg)
        self._times = times
        self._values = values
        self._seed = seed
        self._xp = xp

    # ------------------------------------------------------------------ data

    @property
    def times(self) -> npt.NDArray[np.float64]:
        r"""Grid points :math:`t_0, \dots, t_N` (read-only NumPy array)."""
        return self._times

    @property
    def values(self) -> Any:
        """Values :math:`W(t_n)`, shape ``(n_steps + 1, n_paths, dim)``.

        Immutable for NumPy: the array cannot be made writeable. Other
        backends have no read-only arrays; there the array must not be
        modified in place.
        """
        return self._values

    @property
    def increments(self) -> Any:
        r"""Increments :math:`\Delta W_n = W(t_{n+1}) - W(t_n)`.

        A new array of shape ``(n_steps, n_paths, dim)``.
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
    def namespace(self) -> ModuleType:
        """Array API namespace of the values."""
        return self._xp

    @property
    def dtype(self) -> Any:
        """Dtype of the values."""
        return self._values.dtype

    @property
    def device(self) -> Any:
        """Device of the values."""
        return device_of(self._values)

    @property
    def seed_sequence(self) -> np.random.SeedSequence:
        """Seed from which refinements are derived (a fresh copy on each access)."""
        return frozen_seed_sequence(self._seed)

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
            The coarse path, with its own copy of the values and the same
            seed.

        Raises
        ------
        TypeError
            If `factor` is not an integer.
        ValueError
            If `factor` is not positive or does not divide :attr:`n_steps`.

        Examples
        --------
        >>> fine = WienerProcess().sample_path([0.0, 0.25, 0.5, 0.75, 1.0], seed=1)
        >>> fine.coarsen(2).times
        array([0. , 0.5, 1. ])
        """
        factor = positive_int(factor, "factor")
        if self.n_steps % factor != 0:
            msg = f"factor {factor} does not divide the number of steps {self.n_steps}."
            raise ValueError(msg)
        values = self._xp.asarray(self._values[::factor, ...], copy=True)
        return BrownianPath._from_owned(
            as_time_grid(self._times[::factor], min_points=1),
            values,
            self._seed,
            self._xp,
        )

    def refine(self, factor: int, *, seed: SeedLike = None) -> BrownianPath:
        """Split every step into `factor` equal substeps.

        The new values are sampled from the Brownian bridge between the
        existing ones, so ``path.refine(k).coarsen(k)`` reproduces `path`
        exactly, and ``path.refine(k)`` returns the same path on every call.

        Parameters
        ----------
        factor
            Positive integer.
        seed
            Seed for the new samples. By default they are derived from the
            path's seed and the new times.

        Returns
        -------
        BrownianPath
            The refined path.

        Raises
        ------
        TypeError
            If `factor` is not an integer.
        ValueError
            If `factor` is not positive.

        Examples
        --------
        >>> path = WienerProcess().sample_path([0.0, 1.0], seed=3)
        >>> path.refine(4).n_steps
        4
        >>> bool((path.refine(4).values == path.refine(4).values).all())
        True
        """
        factor = positive_int(factor, "factor")
        fractions = np.arange(1, factor) / factor
        left = self._times[:-1, None]
        new_times = (left + fractions[None, :] * np.diff(self._times)[:, None]).ravel()
        return self.insert(new_times, seed=seed)

    def insert(self, times: TimeGridLike, *, seed: SeedLike = None) -> BrownianPath:
        r"""Add grid points, keeping all values already sampled.

        Points inside the grid are sampled from the Brownian bridge,
        conditioned on the neighboring values; points after :attr:`t_end`
        extend the path with independent increments. Points already on the
        grid are ignored.

        Parameters
        ----------
        times
            New grid points, not smaller than :attr:`t0`; any order.
        seed
            Seed for the new samples. By default they are derived from the
            path's seed and the new times, so the same call always returns
            the same path.

        Returns
        -------
        BrownianPath
            The path on the merged grid, with the same seed as this path. If
            all times are already on the grid, this path itself.

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
        if seed is None:
            stream = derive_seed_sequence(self._seed, new.astype("<f8").tobytes())
        else:
            stream = frozen_seed_sequence(seed)
        rng = RandomNumberGenerator(stream)

        inside = new[new < self.t_end]
        beyond = new[new > self.t_end]
        blocks = [self._values]
        block_times = [self._times]
        if inside.size:
            blocks.append(self._bridge(inside, rng))
            block_times.append(inside)
        if beyond.size:
            blocks.append(self._extend(beyond, rng))
            block_times.append(beyond)

        xp = self._xp
        merged_times = np.concatenate(block_times)
        order = np.argsort(merged_times, kind="stable")
        index = xp.asarray(order, device=array_api_compat.device(self._values))
        merged = xp.take(xp.concat(blocks, axis=0), index, axis=0)
        return BrownianPath._from_owned(
            as_time_grid(merged_times[order], min_points=1), merged, self._seed, xp
        )

    # ---------------------------------------------------------- noise source

    def _steps(self, requires: frozenset[str]) -> Iterator[StepNoise]:
        r"""Yield :math:`h_n` and :math:`\Delta W_n` for every step."""
        times = self._times
        for n in range(self.n_steps):
            yield StepNoise(
                t=float(times[n]),
                h=float(times[n + 1] - times[n]),
                dW=self._values[n + 1, ...] - self._values[n, ...],
            )

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
        seed: SeedLike = None,
        backend: str | ModuleType | None = None,
        dtype: Any = None,
        device: Any = None,
    ) -> BrownianPath:
        r"""Sample independent Brownian paths on a time grid.

        The paths start at :math:`W(t_0) = 0` and have increments
        :math:`\Delta W_n = \sqrt{h_n}\, Z_n`, where :math:`Z_0, Z_1, \dots`
        are the standard normal samples of ``RandomNumberGenerator(seed)``
        in order. The seed is stored in the path for later refinements.

        Parameters
        ----------
        times
            Strictly increasing grid with at least two points.
        n_paths
            Number of independent paths, at least 1.
        seed
            Seed; the same seed gives the same path on every backend.
            ``None`` draws fresh entropy.
        backend
            Array namespace or backend name such as ``"numpy"`` or
            ``"torch"``. Defaults to NumPy.
        dtype
            Real floating-point dtype of the backend; defaults to double
            precision.
        device
            Device of the returned arrays, e.g. ``"cuda"``.

        Returns
        -------
        BrownianPath
            The sampled paths, values of shape ``(len(times), n_paths, dim)``.

        Raises
        ------
        TypeError
            If `n_paths` is not an integer or `dtype` is not a real
            floating-point dtype.
        ValueError
            If the grid is invalid or `n_paths` is not positive.

        Examples
        --------
        >>> from numsde.stepping import uniform_grid
        >>> path = WienerProcess().sample_path(uniform_grid(1.0, 8), n_paths=3, seed=7)
        >>> path.values.shape
        (9, 3, 1)
        """
        grid = as_time_grid(times)
        n_paths = positive_int(n_paths, "n_paths")
        xp = namespace_for(backend) if isinstance(backend, str) else backend
        if xp is None:
            xp = namespace_for("numpy")
        root = frozen_seed_sequence(seed)
        start = BrownianPath._from_owned(
            grid[:1],
            xp.zeros(
                (1, n_paths, self.dim),
                dtype=xp.float64 if dtype is None else dtype,
                device=device,
            ),
            root,
            xp,
        )
        rng = RandomNumberGenerator(frozen_seed_sequence(root))
        values = xp.concat([start.values, start._extend(grid[1:], rng)], axis=0)
        return BrownianPath._from_owned(grid, values, root, xp)
