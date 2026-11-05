r"""Statistics of empirical convergence studies.

The functions work on host NumPy arrays of per-path data and are independent
of the array backend used for the simulation. They implement the steps
described on the theory page :doc:`/theory/convergence_studies`:

1. :func:`mean_interval`: Monte Carlo estimate of a mean with a confidence
   interval from the central limit theorem.
2. :func:`fit_order`: least-squares fit of :math:`\log e_h` against
   :math:`\log h`.
3. :func:`bootstrap_orders` and :func:`jackknife_orders`: resampled orders,
   with the same path indices for all step sizes.
4. :func:`bootstrap_interval`: percentile or BCa interval from the resampled
   values.

Throughout, per-path data have shape ``(..., n_steps, n_paths)``: leading
axes (for example one per scheme), one axis per step size, and the paths
last.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

import numpy as np

from numsde._backend import frozen_seed_sequence
from numsde._validation import positive_int

if TYPE_CHECKING:
    import numpy.typing as npt

    from numsde._backend import SeedLike

__all__ = [
    "CI_METHODS",
    "CIMethod",
    "bootstrap_interval",
    "bootstrap_orders",
    "fit_order",
    "jackknife_orders",
    "mean_interval",
]

CIMethod: TypeAlias = Literal["percentile", "bca"]
"""Names of the bootstrap interval methods."""

CI_METHODS: Final[tuple[str, ...]] = ("percentile", "bca")
"""Supported bootstrap interval methods; the first is the default."""

_STANDARD_NORMAL: Final = NormalDist()

# Resamples drawn per matrix product in bootstrap_orders; bounds the memory
# of the count matrix to about 8 * n_paths * _BATCH bytes.
_BATCH: Final = 64

FloatArray: TypeAlias = "npt.NDArray[np.float64]"


def _check_confidence(confidence: object) -> float:
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        msg = f"confidence must be a real number, got {confidence!r}."
        raise TypeError(msg)
    if not 0.0 < confidence < 1.0:
        msg = f"confidence must lie in (0, 1), got {confidence}."
        raise ValueError(msg)
    return float(confidence)


def mean_interval(
    samples: npt.ArrayLike, confidence: float = 0.95
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    r"""Estimate means with standard errors and normal confidence intervals.

    For samples :math:`Y_1, \dots, Y_n` the estimate is the sample mean
    :math:`\bar Y`, its standard error :math:`s/\sqrt{n}` with the sample
    standard deviation :math:`s`, and the interval
    :math:`\bar Y \pm z_{1-\alpha/2}\, s/\sqrt{n}`, where
    :math:`1 - \alpha` is the confidence level.

    Parameters
    ----------
    samples
        Array of shape ``(..., n)`` with ``n >= 2``; the mean is taken over
        the last axis.
    confidence
        Confidence level in :math:`(0, 1)`.

    Returns
    -------
    mean, standard_error, lower, upper : numpy.ndarray
        Arrays of shape ``samples.shape[:-1]``.

    Raises
    ------
    ValueError
        If fewer than two samples are given or `confidence` is not in
        :math:`(0, 1)`.

    Examples
    --------
    >>> mean, se, lower, upper = mean_interval([1.0, 2.0, 3.0, 4.0])
    >>> float(mean), round(float(se), 4)
    (2.5, 0.6455)
    """
    confidence = _check_confidence(confidence)
    y = np.asarray(samples, dtype=np.float64)
    n = y.shape[-1] if y.ndim else 0
    if n < 2:
        msg = f"At least two samples are needed, got {n}."
        raise ValueError(msg)
    mean = y.mean(axis=-1)
    standard_error = y.std(axis=-1, ddof=1) / math.sqrt(n)
    z = _STANDARD_NORMAL.inv_cdf(0.5 + confidence / 2)
    return mean, standard_error, mean - z * standard_error, mean + z * standard_error


def _fit_weights(
    step_sizes: npt.ArrayLike, mask: npt.ArrayLike | None
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Return w with slope = w @ log(errors), and the mask of fitted step sizes.

    A selected step size can have weight zero (the center of a geometric
    grid), so validity checks use the mask, not the weights.
    """
    h = np.asarray(step_sizes, dtype=np.float64)
    if h.ndim != 1:
        msg = f"step_sizes must be one-dimensional, got shape {h.shape}."
        raise ValueError(msg)
    if np.any(~np.isfinite(h)) or np.any(h <= 0.0):
        msg = "step_sizes must be positive and finite."
        raise ValueError(msg)
    use = np.ones(h.size, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if use.shape != h.shape:
        msg = f"mask must have shape {h.shape}, got {use.shape}."
        raise ValueError(msg)
    x = np.log(h[use])
    if np.unique(x).size < 2:
        msg = "The order fit needs at least two distinct step sizes."
        raise ValueError(msg)
    weights = np.zeros(h.size)
    centered = x - x.mean()
    weights[use] = centered / np.sum(centered**2)
    return weights, use


def fit_order(
    step_sizes: npt.ArrayLike,
    errors: npt.ArrayLike,
    mask: npt.ArrayLike | None = None,
) -> FloatArray:
    r"""Fit the order of convergence by least squares on a log-log scale.

    The order is the slope :math:`\hat q` of the line
    :math:`\log e_k \approx \log C + \hat q \log h_k` that fits the selected
    step sizes best in the least-squares sense.

    Parameters
    ----------
    step_sizes
        Step sizes :math:`h_k`, shape ``(n_steps,)``.
    errors
        Errors :math:`e_k`, shape ``(..., n_steps)``.
    mask
        Boolean array of shape ``(n_steps,)`` selecting the step sizes used
        in the fit; by default all.

    Returns
    -------
    numpy.ndarray
        The fitted orders, shape ``errors.shape[:-1]``. An order is NaN if a
        selected error is zero, as for a scheme that is exact on the problem.

    Raises
    ------
    ValueError
        If the step sizes are not positive, fewer than two distinct step
        sizes are selected, or the shapes do not match.

    Examples
    --------
    >>> h = [0.1, 0.05, 0.025]
    >>> float(fit_order(h, [2 * x**0.5 for x in h]))
    0.5
    """
    weights, use = _fit_weights(step_sizes, mask)
    e = np.asarray(errors, dtype=np.float64)
    if e.shape[-1:] != weights.shape:
        msg = f"errors must have {weights.size} step sizes in the last axis."
        raise ValueError(msg)
    return _slope(weights, use, e)


def _slope(
    weights: FloatArray, selected: npt.NDArray[np.bool_], errors: FloatArray
) -> FloatArray:
    """Slope of the fit for errors of shape (..., n_steps); NaN for zero errors."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_e = np.log(np.where(selected, errors, 1.0))
        slope = np.asarray(log_e @ weights, dtype=np.float64)
    undefined = np.any((errors <= 0.0) & selected, axis=-1)
    return np.where(undefined, np.nan, slope)


def _orders_from_moments(
    moments: FloatArray,
    p: float,
    weights: FloatArray,
    selected: npt.NDArray[np.bool_],
) -> FloatArray:
    """Orders from moments E[Y] of shape (..., n_steps), errors = moments^(1/p)."""
    return _slope(weights, selected, moments ** (1.0 / p))


def bootstrap_orders(
    pathwise: npt.ArrayLike,
    step_sizes: npt.ArrayLike,
    p: float,
    *,
    n_bootstrap: int = 1000,
    mask: npt.ArrayLike | None = None,
    seed: SeedLike = None,
) -> FloatArray:
    r"""Recompute the fitted order for resampled paths.

    Each replicate draws ``n_paths`` path indices with replacement and uses
    them for **all** step sizes, because the errors of different step sizes
    are computed on the same paths and are correlated. From the resampled
    per-path data :math:`Y^*_{ik}` it forms the errors
    :math:`(\overline{Y^*_k})^{1/p}` and fits the order.

    Parameters
    ----------
    pathwise
        Per-path data :math:`Y_{ik}` of shape ``(..., n_steps, n_paths)``,
        for the strong error :math:`Y_{ik} = \max_n |X_{h_k}^n - X(t_n)|^p`.
    step_sizes
        Step sizes, shape ``(n_steps,)``.
    p
        Exponent of the :math:`L^p` norm.
    n_bootstrap
        Number of replicates :math:`B`.
    mask
        Step sizes used in the fit, as for :func:`fit_order`.
    seed
        Seed of the resampling.

    Returns
    -------
    numpy.ndarray
        Replicates of shape ``(..., n_bootstrap)``.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> h = np.array([0.1, 0.05, 0.025])
    >>> y = h[:, None] * rng.lognormal(size=(3, 200))  # order 1/2 for p = 2
    >>> bootstrap_orders(y, h, 2.0, n_bootstrap=100, seed=1).shape
    (100,)
    """
    n_bootstrap = positive_int(n_bootstrap, "n_bootstrap")
    weights, selected = _fit_weights(step_sizes, mask)
    y = _pathwise(pathwise, weights.size)
    lead, n_paths = y.shape[:-1], y.shape[-1]
    flat = y.reshape(-1, n_paths)
    rng = np.random.default_rng(frozen_seed_sequence(seed))
    moments = np.empty((flat.shape[0], n_bootstrap))
    for start in range(0, n_bootstrap, _BATCH):
        stop = min(start + _BATCH, n_bootstrap)
        counts = np.stack(
            [
                np.bincount(rng.integers(0, n_paths, n_paths), minlength=n_paths)
                for _ in range(stop - start)
            ],
            axis=1,
        )
        moments[:, start:stop] = flat @ counts / n_paths
    moments = moments.reshape(*lead, n_bootstrap)
    # Bring the step sizes to the last axis for the fit.
    return _orders_from_moments(np.moveaxis(moments, -2, -1), p, weights, selected)


def jackknife_orders(
    pathwise: npt.ArrayLike,
    step_sizes: npt.ArrayLike,
    p: float,
    *,
    mask: npt.ArrayLike | None = None,
) -> FloatArray:
    r"""Fitted orders with one path left out at a time.

    The leave-one-out means are
    :math:`\bar Y_{(i)} = (n \bar Y - Y_i)/(n - 1)`, so all :math:`n`
    jackknife fits cost about as much as one pass over the data.

    Parameters
    ----------
    pathwise, step_sizes, p, mask
        As for :func:`bootstrap_orders`.

    Returns
    -------
    numpy.ndarray
        Jackknife orders of shape ``(..., n_paths)``.

    Raises
    ------
    ValueError
        If fewer than two paths are given.

    Examples
    --------
    >>> h = np.array([0.1, 0.05])
    >>> y = np.array([[0.1, 0.12, 0.08], [0.05, 0.06, 0.04]])
    >>> jackknife_orders(y, h, 2.0).round(6)
    array([0.5, 0.5, 0.5])
    """
    weights, selected = _fit_weights(step_sizes, mask)
    y = _pathwise(pathwise, weights.size)
    n = y.shape[-1]
    if n < 2:
        msg = f"The jackknife needs at least two paths, got {n}."
        raise ValueError(msg)
    total = y.sum(axis=-1, keepdims=True)
    moments = (total - y) / (n - 1)  # (..., n_steps, n_paths)
    return _orders_from_moments(np.moveaxis(moments, -2, -1), p, weights, selected)


def _pathwise(pathwise: npt.ArrayLike, n_steps: int) -> FloatArray:
    y = np.asarray(pathwise, dtype=np.float64)
    if y.ndim < 2 or y.shape[-2] != n_steps:
        msg = (
            f"pathwise must have shape (..., n_steps, n_paths) with "
            f"n_steps={n_steps}, got {y.shape}."
        )
        raise ValueError(msg)
    return y


def bootstrap_interval(
    estimate: npt.ArrayLike,
    replicates: npt.ArrayLike,
    jackknife: npt.ArrayLike | None = None,
    *,
    confidence: float = 0.95,
    method: CIMethod = "percentile",
) -> tuple[FloatArray, FloatArray]:
    r"""Confidence interval from bootstrap replicates.

    ``"percentile"`` takes the :math:`\alpha/2` and :math:`1 - \alpha/2`
    quantiles of the replicates :math:`\hat\theta^*_b`. ``"bca"``
    (bias-corrected and accelerated) takes the quantiles at the adjusted
    levels

    .. math::

        \alpha_{1,2} = \Phi\Big(\hat z_0 + \frac{\hat z_0 + z_{\alpha/2,\,1-\alpha/2}}
        {1 - \hat a\,(\hat z_0 + z_{\alpha/2,\,1-\alpha/2})}\Big),

    with the bias correction
    :math:`\hat z_0 = \Phi^{-1}\big(\#\{\hat\theta^*_b < \hat\theta\}/B\big)`
    (ties count one half) and the acceleration estimated from the jackknife
    values :math:`\hat\theta_{(i)}` with mean :math:`\hat\theta_{(\cdot)}`,

    .. math::

        \hat a = \frac{\sum_i (\hat\theta_{(\cdot)} - \hat\theta_{(i)})^3}
        {6 \big(\sum_i (\hat\theta_{(\cdot)} - \hat\theta_{(i)})^2\big)^{3/2}}.

    The proportion in :math:`\hat z_0` is clipped to
    :math:`[1/(2B), 1 - 1/(2B)]`, so that :math:`\hat z_0` stays finite.

    Parameters
    ----------
    estimate
        Estimates :math:`\hat\theta` from the full sample, shape ``lead``.
    replicates
        Bootstrap replicates, shape ``lead + (B,)``.
    jackknife
        Jackknife values, shape ``lead + (n,)``; required for ``"bca"``.
    confidence
        Confidence level in :math:`(0, 1)`.
    method
        ``"percentile"`` (default) or ``"bca"``. In simulations of
        convergence studies the percentile interval covered at least as well
        as BCa, see :doc:`/theory/convergence_studies`.

    Returns
    -------
    lower, upper : numpy.ndarray
        Interval limits, shape ``lead``. They are NaN where the estimate, a
        replicate or (for ``"bca"``) a jackknife value is not finite.

    Raises
    ------
    ValueError
        If `method` is unknown, `jackknife` is missing for ``"bca"``, or the
        shapes do not match.

    Examples
    --------
    With symmetric replicates centered at the estimate and no skewness, both
    methods agree:

    >>> reps = np.linspace(-1.0, 1.0, 1001)
    >>> bootstrap_interval(0.0, reps, confidence=0.9)
    (array(-0.9), array(0.9))
    >>> jack = np.array([-1.0, 1.0])
    >>> bootstrap_interval(0.0, reps, jack, confidence=0.9, method="bca")
    (array(-0.9), array(0.9))
    """
    confidence = _check_confidence(confidence)
    if method not in CI_METHODS:
        msg = f"method must be one of {CI_METHODS}, got {method!r}."
        raise ValueError(msg)
    theta = np.asarray(estimate, dtype=np.float64)
    reps = np.asarray(replicates, dtype=np.float64)
    if reps.shape[:-1] != theta.shape or reps.shape[-1] < 1:
        msg = f"replicates must have shape {theta.shape} + (B,), got {reps.shape}."
        raise ValueError(msg)
    jack: FloatArray | None = None
    if method == "bca":
        if jackknife is None:
            msg = "The BCa interval needs the jackknife values."
            raise ValueError(msg)
        jack = np.asarray(jackknife, dtype=np.float64)
        if jack.shape[:-1] != theta.shape or jack.shape[-1] < 2:
            msg = f"jackknife must have shape {theta.shape} + (n,), got {jack.shape}."
            raise ValueError(msg)

    alpha = 1.0 - confidence
    lower = np.full(theta.shape, np.nan)
    upper = np.full(theta.shape, np.nan)
    for index in np.ndindex(theta.shape):
        values = reps[index]
        if not np.isfinite(theta[index]) or not np.all(np.isfinite(values)):
            continue
        if jack is not None and not np.all(np.isfinite(jack[index])):
            # An undefined jackknife value leaves the acceleration undefined.
            continue
        levels = (alpha / 2, 1 - alpha / 2)
        if jack is not None:
            levels = _bca_levels(float(theta[index]), values, jack[index], levels)
        lower[index], upper[index] = np.quantile(values, levels)
    return lower, upper


def _bca_levels(
    theta: float,
    replicates: FloatArray,
    jackknife: FloatArray,
    levels: tuple[float, float],
) -> tuple[float, float]:
    """Return the adjusted quantile levels of the BCa interval."""
    n_rep = replicates.size
    below = np.count_nonzero(replicates < theta)
    ties = np.count_nonzero(replicates == theta)
    raw = float((below + 0.5 * ties) / n_rep)
    proportion = min(max(raw, 0.5 / n_rep), 1.0 - 0.5 / n_rep)
    z0 = _STANDARD_NORMAL.inv_cdf(proportion)
    deviation = jackknife.mean() - jackknife
    spread = float(np.sum(deviation**2))
    a = float(np.sum(deviation**3)) / (6.0 * spread**1.5) if spread > 0.0 else 0.0
    adjusted = []
    for level in levels:
        z = z0 + _STANDARD_NORMAL.inv_cdf(level)
        denominator = 1.0 - a * z
        if denominator <= 0.0:
            # Extreme acceleration: the adjusted level tends to 0 or 1.
            adjusted.append(0.0 if level < 0.5 else 1.0)
        else:
            adjusted.append(_STANDARD_NORMAL.cdf(z0 + z / denominator))
    return adjusted[0], adjusted[1]
