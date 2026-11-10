r"""Empirical strong convergence studies.

A :class:`StrongConvergenceStudy` runs one or more schemes on coarsened
versions of one Brownian path and compares them with a reference solution on
the same path. The strong error

.. math::

    e_h = \Big(\mathbb{E}\Big[\max_{n} \big|X_h^n - X(t_n)\big|^p\Big]\Big)^{1/p}
    \quad\text{or}\quad
    e_h^T = \big(\mathbb{E}\,\big|X_h^N - X(T)\big|^p\big)^{1/p}

is estimated from per-path data, and the order of convergence is fitted with
a pathwise bootstrap interval. See :doc:`/theory/convergence_studies`.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from numsde._backend import frozen_seed_sequence, owned_read_only, to_numpy
from numsde._validation import check_instance, positive_int
from numsde.analysis.statistics import (
    CI_METHODS,
    CIMethod,
    bootstrap_interval,
    bootstrap_orders,
    fit_order,
    jackknife_orders,
    mean_interval,
)
from numsde.noise.brownian import BrownianPath
from numsde.problems.base import SDEProblem
from numsde.schemes.base import Scheme
from numsde.solver import Solver

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes import Axes

    from numsde._backend import SeedLike

__all__ = ["StrongConvergenceResult", "StrongConvergenceStudy"]

ERROR_KINDS = ("grid", "final")


class StrongConvergenceStudy:
    r"""Strong convergence study of one or more schemes on one Brownian path.

    Every scheme runs on ``path.coarsen(k)`` for every factor ``k``, so all
    schemes and step sizes see the same realization of the noise. The
    reference solution is evaluated on the same path:

    * ``reference="exact"`` uses :meth:`SDEProblem.exact_solution
      <numsde.SDEProblem.exact_solution>` at the grid points;
    * a :class:`~numsde.schemes.Scheme` is run on ``path``, refined by the
      Brownian bridge with ``reference_refinement`` substeps per step, and
      restricted to the coarse grids. The reference grid must be strictly
      finer than every grid of the study.

    Parameters
    ----------
    problem
        The problem.
    schemes
        A scheme or a sequence of schemes.
    path
        Brownian path on the finest grid; its dimension must equal
        ``problem.noise_dim`` and its grid must start at ``problem.t0``.
    factors
        Distinct coarsening factors, each dividing ``path.n_steps``; at
        least two. Factor ``k`` gives the step size ``k`` times that of the
        path (for a non-uniform grid, the largest coarse step).
    reference
        ``"exact"`` or a scheme for the reference solution.
    reference_refinement
        Substeps per step of `path` for a scheme reference; must be 1 for
        ``"exact"``.
    p
        Exponent :math:`p \ge 1` of the error norm.
    error
        ``"grid"`` (default) for the maximum over the grid inside the
        expectation, ``"final"`` for the error at the final time.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If the factors, the reference or the error kind are invalid, or the
        path does not match the problem.

    Examples
    --------
    >>> import numsde as ns
    >>> problem = ns.benchmarks.GeometricBrownianMotion(mu=1.0, sigma=1.0)
    >>> path = ns.WienerProcess(1).sample_path(
    ...     ns.uniform_grid(1.0, 2**8), n_paths=2000, seed=1
    ... )
    >>> study = ns.analysis.StrongConvergenceStudy(
    ...     problem, ns.schemes.EulerMaruyama(), path, factors=[2, 4, 8, 16]
    ... )
    >>> result = study.run(n_bootstrap=200, seed=2)
    >>> result.errors.shape, result.orders.shape
    ((1, 4), (1,))
    >>> round(float(result.orders[0]), 2)
    0.48
    """

    def __init__(
        self,
        problem: SDEProblem,
        schemes: Scheme | Sequence[Scheme],
        path: BrownianPath,
        factors: Sequence[int],
        *,
        reference: Literal["exact"] | Scheme = "exact",
        reference_refinement: int = 1,
        p: float = 2.0,
        error: Literal["grid", "final"] = "grid",
    ) -> None:
        check_instance(problem, SDEProblem, "problem")
        self._schemes = _as_schemes(schemes)
        check_instance(path, BrownianPath, "path")
        self._problem = problem
        self._path = path
        self._factors = _as_factors(factors, path.n_steps)
        self._p = _as_exponent(p)
        if error not in ERROR_KINDS:
            msg = f"error must be one of {ERROR_KINDS}, got {error!r}."
            raise ValueError(msg)
        self._error = error
        self._refinement = positive_int(reference_refinement, "reference_refinement")
        given: object = reference
        if isinstance(given, Scheme):
            if self._factors[0] * self._refinement < 2:
                msg = (
                    "The reference grid must be finer than every grid of the "
                    "study: use factors > 1 or reference_refinement > 1."
                )
                raise ValueError(msg)
            self._reference: Scheme | None = given
        elif given == "exact":
            if not problem.has_exact_solution:
                msg = (
                    f"{type(problem).__name__} has no exact solution; pass a "
                    "scheme as reference."
                )
                raise ValueError(msg)
            if self._refinement != 1:
                msg = "reference_refinement applies only to a scheme reference."
                raise ValueError(msg)
            self._reference = None
        else:
            msg = f"reference must be 'exact' or a Scheme, got {given!r}."
            raise TypeError(msg)
        # Check schemes, reference and path against the problem now, not after
        # the first simulations: iterate() runs all checks without stepping.
        self._solvers = [Solver(problem, scheme) for scheme in self._schemes]
        probe = path.coarsen(path.n_steps)
        checked = list(self._solvers)
        if self._reference is not None:
            checked.append(Solver(problem, self._reference))
        for solver in checked:
            solver.iterate(probe)

    # ------------------------------------------------------------ properties

    @property
    def problem(self) -> SDEProblem:
        """The problem."""
        return self._problem

    @property
    def schemes(self) -> tuple[Scheme, ...]:
        """The schemes under study."""
        return self._schemes

    @property
    def factors(self) -> tuple[int, ...]:
        """Coarsening factors, finest step size first."""
        return self._factors

    @property
    def step_sizes(self) -> npt.NDArray[np.float64]:
        """Largest step size of each coarse grid, finest first."""
        return np.array(
            [np.max(self._path.coarsen(k).step_sizes) for k in self._factors]
        )

    # ------------------------------------------------------------ simulation

    def simulate(self) -> npt.NDArray[np.float64]:
        r"""Compute the per-path data of the study.

        Returns
        -------
        numpy.ndarray
            :math:`Y_{s,k,i} = \max_n |X_{h_k}^n - X(t_n)|^p` (or the error
            at the final time) of scheme ``s``, factor ``k`` and path ``i``;
            shape ``(n_schemes, n_factors, n_paths)``.
        """
        reference = self._reference_values()
        data = np.empty((len(self._schemes), len(self._factors), self._path.n_paths))
        for s, solver in enumerate(self._solvers):
            for k, factor in enumerate(self._factors):
                coarse = self._path.coarsen(factor)
                data[s, k] = self._pathwise(solver, coarse, factor, reference)
        return data

    def _needed_indices(self) -> set[int]:
        """Return the indices on the grid of `path` at which errors are taken.

        These are the points of all coarse grids, which need not be nested,
        e.g. for the factors 2 and 3; for the final error only the last point.
        """
        n_steps = self._path.n_steps
        if self._error == "final":
            return {n_steps}
        return {n for k in self._factors for n in range(0, n_steps + 1, k)}

    def _reference_values(self) -> dict[int, Any] | None:
        """Return a scheme reference by grid index of `path`; ``None`` if exact.

        The reference runs on `path` refined by the Brownian bridge; its point
        ``m * reference_refinement`` is the point ``m`` of `path`. Only the
        points needed by some coarse grid are kept.
        """
        if self._reference is None:
            return None
        needed = self._needed_indices()
        r = self._refinement
        solver = Solver(self._problem, self._reference)
        return {
            m // r: x
            for m, (_, x) in enumerate(solver.iterate(self._path.refine(r)))
            if m % r == 0 and m // r in needed
        }

    def _pathwise(
        self,
        solver: Solver,
        coarse: BrownianPath,
        factor: int,
        reference: dict[int, Any] | None,
    ) -> npt.NDArray[np.float64]:
        """Per-path error of one scheme on one coarse path, as a host array."""
        xp = coarse.namespace
        problem = self._problem
        worst: Any = None
        last = coarse.n_steps
        for n, (t, x) in enumerate(solver.iterate(coarse)):
            if self._error == "final" and n < last:
                continue
            if reference is None:
                W = coarse.values[n, ...] - coarse.values[0, ...]
                exact = problem.exact_solution(t, W)
            else:
                exact = reference[n * factor]  # index on the grid of `path`
            diff = x - exact
            err = xp.sum(diff * diff, axis=-1) ** (self._p / 2)
            worst = err if worst is None else xp.maximum(worst, err)
        return to_numpy(worst).astype(np.float64)

    def run(
        self,
        *,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        method: CIMethod = "percentile",
        seed: SeedLike = None,
        fit_factors: Sequence[int] | None = None,
    ) -> StrongConvergenceResult:
        """Simulate and analyze the study.

        Parameters
        ----------
        n_bootstrap
            Number of bootstrap replicates for the interval of the order.
        confidence
            Confidence level of all intervals.
        method
            ``"percentile"`` (default) or ``"bca"`` bootstrap interval.
        seed
            Seed of the bootstrap resampling; the Brownian path carries its
            own seed.
        fit_factors
            Factors whose step sizes enter the order fit; by default all.

        Returns
        -------
        StrongConvergenceResult
            Errors, intervals and fitted orders.
        """
        return StrongConvergenceResult.from_pathwise(
            self.simulate(),
            step_sizes=self.step_sizes,
            factors=self._factors,
            scheme_names=tuple(s.name for s in self._schemes),
            expected_orders=tuple(
                s.expected_strong_order(self._problem.structure) for s in self._schemes
            ),
            p=self._p,
            error=self._error,
            reference=self._describe_reference(),
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            method=method,
            seed=seed,
            fit_factors=fit_factors,
        )

    def _describe_reference(self) -> str:
        if self._reference is None:
            return "exact solution"
        n_steps = self._path.n_steps * self._refinement
        return f"{self._reference.name} with {n_steps} steps"

    def __repr__(self) -> str:
        """Return a summary of the study."""
        names = ", ".join(s.name for s in self._schemes)
        return (
            f"StrongConvergenceStudy(schemes=[{names}], factors={list(self._factors)}, "
            f"n_paths={self._path.n_paths}, p={self._p}, error={self._error!r}, "
            f"reference={self._describe_reference()!r})"
        )


@dataclass(frozen=True, eq=False)
class StrongConvergenceResult:
    r"""Result of a :class:`StrongConvergenceStudy`.

    Arrays have a leading axis over the schemes. Create results with
    :meth:`StrongConvergenceStudy.run`, or with :meth:`from_pathwise` from
    per-path data; change the statistical settings with :meth:`refit`
    without simulating again.

    Attributes
    ----------
    scheme_names : tuple of str
        Names of the schemes.
    factors : numpy.ndarray
        Coarsening factors, finest first.
    step_sizes : numpy.ndarray
        Step sizes :math:`h_k`.
    pathwise : numpy.ndarray
        Per-path data, shape ``(n_schemes, n_factors, n_paths)``.
    errors : numpy.ndarray
        Estimated errors :math:`e_{h_k}`, shape ``(n_schemes, n_factors)``.
    errors_ci : numpy.ndarray
        Confidence intervals of the errors, shape ``(n_schemes, n_factors, 2)``.
    orders : numpy.ndarray
        Fitted orders, shape ``(n_schemes,)``.
    orders_ci : numpy.ndarray
        Bootstrap intervals of the orders, shape ``(n_schemes, 2)``.
    expected_orders : numpy.ndarray
        Orders expected from theory for the problem's noise structure.
    fit_mask : numpy.ndarray
        Which step sizes entered the fit.
    """

    scheme_names: tuple[str, ...]
    factors: npt.NDArray[np.int64]
    step_sizes: npt.NDArray[np.float64]
    pathwise: npt.NDArray[np.float64] = field(repr=False)
    errors: npt.NDArray[np.float64]
    errors_ci: npt.NDArray[np.float64]
    orders: npt.NDArray[np.float64]
    orders_ci: npt.NDArray[np.float64]
    expected_orders: npt.NDArray[np.float64]
    fit_mask: npt.NDArray[np.bool_]
    p: float
    error: str
    reference: str
    n_bootstrap: int
    confidence: float
    method: str
    seed: np.random.SeedSequence = field(repr=False)

    @classmethod
    def from_pathwise(
        cls,
        pathwise: npt.ArrayLike,
        *,
        step_sizes: npt.ArrayLike,
        factors: Sequence[int],
        scheme_names: Sequence[str],
        expected_orders: Sequence[float],
        p: float,
        error: str,
        reference: str,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        method: CIMethod = "percentile",
        seed: SeedLike = None,
        fit_factors: Sequence[int] | None = None,
    ) -> StrongConvergenceResult:
        r"""Analyze per-path data.

        Parameters
        ----------
        pathwise
            Per-path data :math:`Y`, shape ``(n_schemes, n_factors, n_paths)``.
        step_sizes, factors, scheme_names, expected_orders
            Description of the axes of `pathwise`.
        p, error, reference
            Exponent, error kind and description of the reference.
        n_bootstrap, confidence, method, seed, fit_factors
            As for :meth:`StrongConvergenceStudy.run`.

        Returns
        -------
        StrongConvergenceResult
            The analyzed result.

        Raises
        ------
        ValueError
            If the shapes do not match, `method` is unknown, or the fit
            selects fewer than two step sizes.

        Examples
        --------
        >>> import numpy as np
        >>> h = np.array([0.1, 0.05, 0.025])
        >>> y = h[None, :, None] * np.random.default_rng(0).lognormal(size=(1, 3, 500))
        >>> result = StrongConvergenceResult.from_pathwise(
        ...     y, step_sizes=h, factors=[4, 2, 1], scheme_names=["demo"],
        ...     expected_orders=[0.5], p=2.0, error="grid", reference="exact",
        ...     n_bootstrap=200, seed=0,
        ... )
        >>> round(float(result.orders[0]), 2)
        0.5
        """
        y = np.array(pathwise, dtype=np.float64)
        h = np.array(step_sizes, dtype=np.float64)
        factor_array = np.array(factors, dtype=np.int64)
        names = tuple(scheme_names)
        expected = np.array(expected_orders, dtype=np.float64)
        if y.ndim != 3 or y.shape[:2] != (len(names), h.size):
            msg = (
                "pathwise must have shape (n_schemes, n_factors, n_paths) = "
                f"({len(names)}, {h.size}, n_paths), got {y.shape}."
            )
            raise ValueError(msg)
        if factor_array.shape != h.shape or expected.shape != (len(names),):
            msg = "factors, step_sizes and expected_orders do not match pathwise."
            raise ValueError(msg)
        if method not in CI_METHODS:
            msg = f"method must be one of {CI_METHODS}, got {method!r}."
            raise ValueError(msg)
        n_bootstrap = positive_int(n_bootstrap, "n_bootstrap")
        if fit_factors is None:
            mask = np.ones(h.size, dtype=bool)
        else:
            chosen = {int(k) for k in fit_factors}
            unknown = chosen - set(factor_array.tolist())
            if unknown:
                msg = f"fit_factors {sorted(unknown)} are not factors of the study."
                raise ValueError(msg)
            mask = np.isin(factor_array, sorted(chosen))
        seed_sequence = frozen_seed_sequence(seed)

        moments, _, lower, upper = mean_interval(y, confidence)
        exponent = 1.0 / p
        errors = moments**exponent
        errors_ci = np.stack(
            [np.maximum(lower, 0.0) ** exponent, upper**exponent], axis=-1
        )
        orders = fit_order(h, errors, mask)
        replicates = bootstrap_orders(
            y, h, p, n_bootstrap=n_bootstrap, mask=mask, seed=seed_sequence
        )
        jackknife = jackknife_orders(y, h, p, mask=mask) if method == "bca" else None
        orders_lower, orders_upper = bootstrap_interval(
            orders, replicates, jackknife, confidence=confidence, method=method
        )

        def frozen(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
            return owned_read_only(array, np, copy=False)  # type: ignore[no-any-return]

        return cls(
            scheme_names=names,
            factors=frozen(factor_array),
            step_sizes=frozen(h),
            pathwise=frozen(y),
            errors=frozen(errors),
            errors_ci=frozen(errors_ci),
            orders=frozen(np.asarray(orders)),
            orders_ci=frozen(np.stack([orders_lower, orders_upper], axis=-1)),
            expected_orders=frozen(expected),
            fit_mask=frozen(mask),
            p=float(p),
            error=error,
            reference=reference,
            n_bootstrap=n_bootstrap,
            confidence=float(confidence),
            method=method,
            seed=seed_sequence,
        )

    @property
    def n_paths(self) -> int:
        """Number of paths."""
        return int(self.pathwise.shape[-1])

    @property
    def eoc(self) -> npt.NDArray[np.float64]:
        r"""Experimental orders between consecutive step sizes.

        :math:`\log(e_{k+1}/e_k) / \log(h_{k+1}/h_k)`, shape
        ``(n_schemes, n_factors - 1)``.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.diff(np.log(self.errors), axis=-1) / np.diff(
                np.log(self.step_sizes)
            )

    def refit(
        self,
        *,
        fit_factors: Sequence[int] | None = None,
        n_bootstrap: int | None = None,
        confidence: float | None = None,
        method: CIMethod | None = None,
        seed: SeedLike = None,
    ) -> StrongConvergenceResult:
        """Analyze the same per-path data with other statistical settings.

        Parameters
        ----------
        fit_factors
            Factors whose step sizes enter the fit; by default all.
        n_bootstrap, confidence, method
            New settings; by default those of this result.
        seed
            Seed of the resampling; by default the seed of this result, so
            that only the changed settings change the result.

        Returns
        -------
        StrongConvergenceResult
            The new result; this result is unchanged.

        Examples
        --------
        >>> import numpy as np
        >>> h = np.array([0.1, 0.05, 0.025])
        >>> y = h[None, :, None] * np.random.default_rng(0).lognormal(size=(1, 3, 500))
        >>> result = StrongConvergenceResult.from_pathwise(
        ...     y, step_sizes=h, factors=[4, 2, 1], scheme_names=["demo"],
        ...     expected_orders=[0.5], p=2.0, error="grid", reference="exact",
        ...     n_bootstrap=200, seed=0,
        ... )
        >>> result.refit(fit_factors=[2, 1]).fit_mask
        array([False,  True,  True])
        """
        return StrongConvergenceResult.from_pathwise(
            self.pathwise,
            step_sizes=self.step_sizes,
            factors=self.factors.tolist(),
            scheme_names=self.scheme_names,
            expected_orders=self.expected_orders.tolist(),
            p=self.p,
            error=self.error,
            reference=self.reference,
            n_bootstrap=self.n_bootstrap if n_bootstrap is None else n_bootstrap,
            confidence=self.confidence if confidence is None else confidence,
            method=self.method if method is None else method,  # type: ignore[arg-type]
            seed=self.seed if seed is None else seed,
            fit_factors=fit_factors,
        )

    def format(self) -> str:
        """Return the result as a text table, one block per scheme."""
        kind = "uniformly on the grid" if self.error == "grid" else "at the final time"
        level = f"{100 * self.confidence:g}%"
        method = "BCa" if self.method == "bca" else "percentile"
        lines = [
            f"Strong error {kind}, p = {self.p:g}, reference: {self.reference}, "
            f"{self.n_paths} paths",
        ]
        for s, name in enumerate(self.scheme_names):
            lines += [
                "",
                f"{name} (expected order {self.expected_orders[s]:g})",
                f"{'h':>12} {'error':>12}   {level + ' CI':<27} {'EOC':>6}",
            ]
            eoc = [np.nan, *self.eoc[s]]
            for k, h in enumerate(self.step_sizes):
                lo, up = self.errors_ci[s, k]
                order = "" if np.isnan(eoc[k]) else f"{eoc[k]:6.2f}"
                note = "" if self.fit_mask[k] else "  (not in fit)"
                lines.append(
                    f"{h:12.4e} {self.errors[s, k]:12.4e}   "
                    f"[{lo:.4e}, {up:.4e}]  {order:>6}{note}"
                )
            lo, up = self.orders_ci[s]
            lines.append(
                f"fitted order {self.orders[s]:.3f}, {level} CI [{lo:.3f}, {up:.3f}] "
                f"({method}, {self.n_bootstrap} resamples)"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return :meth:`format`."""
        return self.format()

    def plot(self, ax: Axes | None = None) -> Axes:
        """Plot the errors against the step sizes on a log-log scale.

        Error bars show the confidence intervals of the errors; dashed lines
        show the fitted orders through the fitted step sizes. Requires
        matplotlib (``pip install 'numsde[plot]'``).

        Parameters
        ----------
        ax
            Axes to draw into; by default a new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        h = self.step_sizes
        for s, name in enumerate(self.scheme_names):
            e = self.errors[s]
            yerr = np.abs(self.errors_ci[s].T - e)
            ax.errorbar(h, e, yerr=yerr, ls="none", marker="o", label=name)
            used = self.fit_mask
            log_c = np.mean(np.log(e[used]) - self.orders[s] * np.log(h[used]))
            ax.plot(
                h[used],
                np.exp(log_c) * h[used] ** self.orders[s],
                "--",
                color="gray",
                label=f"fitted order {self.orders[s]:.2f}",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("step size $h$")
        ax.set_ylabel("strong error")
        ax.legend()
        return ax


def _as_schemes(schemes: object) -> tuple[Scheme, ...]:
    if isinstance(schemes, Scheme):
        return (schemes,)
    if not isinstance(schemes, Sequence) or isinstance(schemes, str):
        msg = f"schemes must be a Scheme or a sequence of schemes, got {schemes!r}."
        raise TypeError(msg)
    result = tuple(schemes)
    if not result:
        msg = "At least one scheme is needed."
        raise ValueError(msg)
    for scheme in result:
        if not isinstance(scheme, Scheme):
            msg = f"schemes must contain Scheme instances, got {scheme!r}."
            raise TypeError(msg)
    return result


def _as_factors(factors: object, n_steps: int) -> tuple[int, ...]:
    if not isinstance(factors, Sequence) or isinstance(factors, str):
        msg = f"factors must be a sequence of integers, got {factors!r}."
        raise TypeError(msg)
    values = [positive_int(k, "factors") for k in factors]
    if len(set(values)) < 2 or len(set(values)) != len(values):
        msg = f"factors must contain at least two distinct values, got {values}."
        raise ValueError(msg)
    for k in values:
        if n_steps % k:
            msg = f"factor {k} does not divide the number of steps {n_steps}."
            raise ValueError(msg)
    return tuple(sorted(values))


def _as_exponent(p: object) -> float:
    if isinstance(p, bool) or not isinstance(p, numbers.Real):
        msg = f"p must be a real number, got {p!r}."
        raise TypeError(msg)
    if not 1.0 <= float(p) < np.inf:
        msg = f"p must be finite and at least 1, got {p}."
        raise ValueError(msg)
    return float(p)
