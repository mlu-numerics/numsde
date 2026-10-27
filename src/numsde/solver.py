"""The solver: the only place where problem, scheme and noise meet."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from numsde._backend import (
    array_namespace,
    device_of,
    namespace_for,
    owned_read_only,
)
from numsde._validation import positive_int
from numsde.noise.brownian import WienerProcess
from numsde.noise.steps import NoiseSource
from numsde.problems.base import SDEProblem, resolve_structure
from numsde.schemes.base import Scheme
from numsde.solution import Solution

if TYPE_CHECKING:
    from types import ModuleType

    from numsde._backend import SeedLike
    from numsde.stepping.grids import TimeGridLike

__all__ = ["Solver"]

_SAVE_MODES = ("all", "final")


class Solver:
    """Run a scheme on a problem over a batch of paths.

    Before the time loop starts, the solver checks that the scheme applies to
    the problem (calculus, noise structure), that the noise source provides
    the step quantities the scheme needs, and that the coefficients return
    arrays of the documented shapes. Problems are reported with a readable
    message instead of a failure deep inside the loop.

    Parameters
    ----------
    problem
        The initial value problem.
    scheme
        The scheme.

    Raises
    ------
    TypeError
        If `problem` or `scheme` has the wrong type.
    ValueError
        If the scheme does not apply to the problem.

    Examples
    --------
    >>> import numsde as ns
    >>> problem = ns.benchmarks.GeometricBrownianMotion()
    >>> path = ns.WienerProcess(1).sample_path(
    ...     ns.uniform_grid(1.0, 2**6), n_paths=100, seed=1
    ... )
    >>> solver = ns.Solver(problem, ns.schemes.EulerMaruyama())
    >>> fine, coarse = solver.solve(path), solver.solve(path.coarsen(4))
    >>> fine.t.size, coarse.t.size
    (65, 17)
    """

    def __init__(self, problem: SDEProblem, scheme: Scheme) -> None:
        for name, value, cls in (
            ("problem", problem, SDEProblem),
            ("scheme", scheme, Scheme),
        ):
            if not isinstance(value, cls):
                msg = (
                    f"{name} must be an instance of {cls.__name__}, "
                    f"got {type(value).__name__}."
                )
                raise TypeError(msg)
        scheme.check_problem(problem)
        self._problem = problem
        self._scheme = scheme

    @property
    def problem(self) -> SDEProblem:
        """The problem."""
        return self._problem

    @property
    def scheme(self) -> Scheme:
        """The scheme."""
        return self._scheme

    def solve(
        self,
        noise: NoiseSource | None = None,
        *,
        times: TimeGridLike | None = None,
        n_paths: int | None = None,
        seed: SeedLike = None,
        backend: str | ModuleType | None = None,
        dtype: Any = None,
        device: Any = None,
        save: Literal["all", "final"] = "all",
    ) -> Solution:
        """Approximate the solution on the grid of a noise realization.

        Either pass a noise source, typically a
        :class:`~numsde.BrownianPath`, or let the solver sample a Wiener
        path on `times`. Passing the same source to several runs gives
        coupled results: different schemes, or different step sizes via
        :meth:`BrownianPath.coarsen <numsde.BrownianPath.coarsen>`, see the
        same realization.

        Parameters
        ----------
        noise
            Noise source whose grid starts at the problem's initial time and
            whose dimension equals ``problem.noise_dim``. The simulation runs
            in the namespace, dtype and device of the source; if ``x0`` is an
            array, they must agree with it.
        times
            Time grid on which to sample a new Wiener path; only without
            `noise`.
        n_paths
            Number of paths to sample (default 1); only without `noise`.
        seed
            Seed of the sampled path; only without `noise`.
        backend, dtype, device
            Namespace (or backend name), dtype and device of a sampled path;
            only without `noise`. They default to those of ``x0`` if it is an
            array, otherwise to NumPy and float64.
        save
            ``"all"`` stores every grid point, ``"final"`` only the last one.

        Returns
        -------
        Solution
            The approximation.

        Raises
        ------
        TypeError
            If `noise` is not a noise source, or sampling options are
            combined with `noise`.
        ValueError
            If neither `noise` nor `times` is given, the noise does not match
            the problem, the source cannot provide the required step
            quantities, or a coefficient returns an array of the wrong shape.
        """
        if save not in _SAVE_MODES:
            msg = f"save must be one of {_SAVE_MODES}, got {save!r}."
            raise ValueError(msg)
        if noise is None:
            noise = self._sample_noise(times, n_paths, seed, backend, dtype, device)
        else:
            self._check_noise(noise, times, n_paths, seed, backend, dtype, device)
        return self._run(noise, save)

    # ------------------------------------------------------------- set-up

    def _sample_noise(
        self,
        times: TimeGridLike | None,
        n_paths: int | None,
        seed: SeedLike,
        backend: str | ModuleType | None,
        dtype: Any,
        device: Any,
    ) -> NoiseSource:
        if times is None:
            msg = "Pass a noise source or the times on which to sample one."
            raise ValueError(msg)
        problem = self._problem
        xp: ModuleType
        if backend is None and problem.x0_is_array:
            xp = array_namespace(problem.x0)
            dtype = problem.x0.dtype if dtype is None else dtype
            device = device_of(problem.x0) if device is None else device
        elif isinstance(backend, str):
            xp = namespace_for(backend)
        else:
            xp = namespace_for("numpy") if backend is None else backend
        return WienerProcess(problem.noise_dim).sample_path(
            times,
            1 if n_paths is None else positive_int(n_paths, "n_paths"),
            seed=seed,
            backend=xp,
            dtype=dtype,
            device=device,
        )

    def _check_noise(
        self,
        noise: object,
        times: object,
        n_paths: object,
        seed: object,
        backend: object,
        dtype: object,
        device: object,
    ) -> None:
        if not isinstance(noise, NoiseSource):
            msg = f"noise must be a NoiseSource, got {type(noise).__name__}."
            raise TypeError(msg)
        options = {
            "times": times,
            "n_paths": n_paths,
            "seed": seed,
            "backend": backend,
            "dtype": dtype,
            "device": device,
        }
        given = [name for name, value in options.items() if value is not None]
        if given:
            msg = (
                f"The sampling options {given} cannot be combined with a noise "
                "source; they only apply when the solver samples the noise."
            )
            raise TypeError(msg)
        problem = self._problem
        if noise.dim != problem.noise_dim:
            msg = (
                f"The noise has dimension {noise.dim}, but the problem needs "
                f"noise_dim={problem.noise_dim}."
            )
            raise ValueError(msg)
        if float(noise.times[0]) != problem.t0:
            msg = (
                f"The noise grid starts at {float(noise.times[0])}, but the "
                f"problem's initial time is t0={problem.t0}."
            )
            raise ValueError(msg)
        if problem.x0_is_array:
            x0 = problem.x0
            if array_namespace(x0) is not noise.namespace:
                msg = (
                    "x0 and the noise source use different array libraries "
                    f"({array_namespace(x0).__name__} and "
                    f"{noise.namespace.__name__})."
                )
                raise ValueError(msg)
            if x0.dtype != noise.dtype or device_of(x0) != noise.device:
                msg = (
                    f"x0 ({x0.dtype} on {device_of(x0)}) and the noise source "
                    f"({noise.dtype} on {noise.device}) differ in dtype or device."
                )
                raise ValueError(msg)

    def _check_coefficients(self, x: Any) -> None:
        problem = self._problem
        n_paths, d, m = x.shape[0], problem.dim, problem.noise_dim
        checks = (
            ("drift", problem.drift, (n_paths, d), "(n_paths, dim)"),
            ("diffusion", problem.diffusion, (n_paths, d, m), "(n_paths, dim, m)"),
        )
        for name, method, shape, label in checks:
            value = method(problem.t0, x)
            got = tuple(getattr(value, "shape", ()))
            if got != shape:
                msg = (
                    f"{type(problem).__name__}.{name} must return an array of "
                    f"shape {label} = {shape} for x of shape {tuple(x.shape)}, "
                    f"got {got}."
                )
                raise ValueError(msg)

    # ------------------------------------------------------------ time loop

    def _run(self, noise: NoiseSource, save: str) -> Solution:
        problem, scheme = self._problem, self._scheme
        noise.check_requires(scheme.requires)
        structure = resolve_structure(problem.structure, problem.dim, problem.noise_dim)
        xp = noise.namespace
        x = problem.initial_value(
            noise.n_paths, xp=xp, dtype=noise.dtype, device=noise.device
        )
        self._check_coefficients(x)
        state = scheme.init_state(problem, x)

        saved = [x]
        for step in noise.steps(scheme.requires):
            x = scheme.step(problem, step.t, x, step, state)
            if save == "all":
                saved.append(x)
        if save == "final":
            saved = [x]
            t = noise.times[-1:]
        else:
            t = noise.times
        metadata = {
            "scheme": scheme.name,
            "n_steps": noise.n_steps,
            "n_paths": noise.n_paths,
            "backend": xp.__name__,
            "dtype": str(noise.dtype),
            "device": str(noise.device),
            "save": save,
            "structure": structure.flags,
            "verified": structure.verified,
        }
        return Solution(
            owned_read_only(np.array(t, dtype=np.float64), np, copy=False),
            xp.stack(saved, axis=0),
            problem=problem,
            scheme=scheme,
            metadata=metadata,
        )
