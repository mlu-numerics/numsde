import math
from statistics import NormalDist

import numpy as np
import pytest

from numsde.analysis.statistics import (
    CI_METHODS,
    bootstrap_interval,
    bootstrap_orders,
    fit_order,
    jackknife_orders,
    mean_interval,
)

H = 2.0 ** -np.arange(2, 6)


def power_law_data(n=200, scale=1.2, seed=2026):
    """Per-path data with population order 1/2 for p = 2, correlated across h."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=n)
    gk = rng.normal(size=(H.size, n))
    return H[:, None] * np.exp(scale * (0.8 * g + 0.6 * gk))


# -------------------------------------------------------------- mean_interval


def test_mean_interval_by_hand():
    mean, se, lower, upper = mean_interval([1.0, 2.0, 3.0, 4.0], confidence=0.9)
    assert mean == 2.5
    assert se == pytest.approx(math.sqrt(5 / 3) / 2)
    z = NormalDist().inv_cdf(0.95)
    assert (lower, upper) == pytest.approx((2.5 - z * se, 2.5 + z * se))


def test_mean_interval_over_the_last_axis():
    y = np.arange(12.0).reshape(2, 2, 3)
    mean, se, lower, upper = mean_interval(y)
    assert mean.shape == se.shape == lower.shape == upper.shape == (2, 2)
    np.testing.assert_allclose(mean, y.mean(-1))


@pytest.mark.parametrize("samples", [[1.0], 3.0, []])
def test_mean_interval_needs_two_samples(samples):
    with pytest.raises(ValueError, match="two samples"):
        mean_interval(samples)


@pytest.mark.parametrize(
    ("confidence", "error"),
    [
        (0.0, ValueError),
        (1.0, ValueError),
        (95, ValueError),
        (True, TypeError),
        ("0.9", TypeError),
    ],
)
def test_confidence_is_validated(confidence, error):
    with pytest.raises(error, match="confidence"):
        mean_interval([1.0, 2.0], confidence=confidence)


# -------------------------------------------------------------------- fit_order


def test_fit_order_of_an_exact_power_law():
    errors = 3.0 * H**0.75
    assert float(fit_order(H, errors)) == pytest.approx(0.75)


def test_fit_order_is_a_least_squares_slope():
    rng = np.random.default_rng(1)
    errors = H**0.5 * np.exp(0.1 * rng.normal(size=H.size))
    expected = np.polyfit(np.log(H), np.log(errors), 1)[0]
    assert float(fit_order(H, errors)) == pytest.approx(expected)


def test_fit_order_with_mask_and_leading_axes():
    errors = np.stack([H**0.5, H**1.0])
    errors[:, 0] *= 10.0  # pre-asymptotic coarsest step size
    mask = [False, True, True, True]
    np.testing.assert_allclose(fit_order(H, errors, mask), [0.5, 1.0])
    assert np.all(fit_order(H, errors) != fit_order(H, errors, mask))


def test_fit_order_is_nan_for_zero_errors():
    errors = np.stack([np.zeros(H.size), H])
    orders = fit_order(H, errors)
    assert np.isnan(orders[0])
    assert orders[1] == pytest.approx(1.0)
    # A zero error outside the fit does not matter.
    with_zero = H.copy()
    with_zero[0] = 0.0
    assert float(fit_order(H, with_zero, [False, True, True, True])) == pytest.approx(
        1.0
    )


@pytest.mark.parametrize(
    ("step_sizes", "mask", "match"),
    [
        (np.ones((2, 2)), None, "one-dimensional"),
        ([0.1, 0.0], None, "positive"),
        ([0.1, np.inf], None, "positive"),
        ([0.1, 0.1], None, "two distinct"),
        ([0.1, 0.05, 0.025], [True, False, False], "two distinct"),
        ([0.1, 0.05], [True], "mask must have shape"),
    ],
)
def test_fit_order_rejects_bad_step_sizes(step_sizes, mask, match):
    errors = np.ones(np.asarray(step_sizes).shape[-1])
    with pytest.raises(ValueError, match=match):
        fit_order(step_sizes, errors, mask)


def test_fit_order_rejects_mismatched_errors():
    with pytest.raises(ValueError, match="4 step sizes"):
        fit_order(H, np.ones(3))


# ----------------------------------------------------- bootstrap and jackknife


def test_bootstrap_uses_the_same_paths_for_all_step_sizes():
    # Y_ik = h_k * Z_i: every resample has the same order, but only if the
    # same path indices are used for all step sizes.
    z = np.random.default_rng(3).lognormal(sigma=1.5, size=300)
    y = H[:, None] * z[None, :]
    reps = bootstrap_orders(y, H, 2.0, n_bootstrap=200, seed=4)
    np.testing.assert_allclose(reps, 0.5, rtol=1e-12)


def test_bootstrap_is_reproducible_and_batched():
    y = power_law_data()
    a = bootstrap_orders(y, H, 2.0, n_bootstrap=70, seed=5)  # crosses a batch
    b = bootstrap_orders(y, H, 2.0, n_bootstrap=70, seed=5)
    assert a.shape == (70,)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, bootstrap_orders(y, H, 2.0, n_bootstrap=70, seed=6))


def test_bootstrap_with_leading_axes_and_mask():
    y = np.stack([power_law_data(seed=1), power_law_data(seed=2)])
    reps = bootstrap_orders(y, H, 2.0, n_bootstrap=50, seed=0, mask=[0, 1, 1, 1])
    assert reps.shape == (2, 50)
    assert np.all(np.isfinite(reps))


def test_bootstrap_rejects_bad_input():
    with pytest.raises(ValueError, match="n_steps=4"):
        bootstrap_orders(np.ones((3, 10)), H, 2.0, n_bootstrap=5)
    with pytest.raises(ValueError, match="n_bootstrap"):
        bootstrap_orders(np.ones((4, 10)), H, 2.0, n_bootstrap=0)


def test_jackknife_equals_leaving_out_each_path():
    y = power_law_data(n=25)
    jack = jackknife_orders(y, H, 2.0)
    brute = [
        float(fit_order(H, np.delete(y, i, axis=-1).mean(-1) ** 0.5))
        for i in range(y.shape[-1])
    ]
    np.testing.assert_allclose(jack, brute, rtol=1e-12)


def test_jackknife_needs_two_paths():
    with pytest.raises(ValueError, match="at least two paths"):
        jackknife_orders(np.ones((4, 1)), H, 2.0)


# --------------------------------------------------------- bootstrap_interval


def test_methods():
    assert CI_METHODS == ("percentile", "bca")


def test_percentile_interval_is_a_pair_of_quantiles():
    reps = np.random.default_rng(0).normal(size=1000)
    lower, upper = bootstrap_interval(0.1, reps, confidence=0.8)
    np.testing.assert_allclose([lower, upper], np.quantile(reps, [0.1, 0.9]))


def test_intervals_match_scipy():
    # Reference values from scipy.stats.bootstrap (SciPy 1.18.1), given the
    # same replicates through `bootstrap_result`; see the pull request that
    # introduced this module.
    y = power_law_data()
    theta = fit_order(H, y.mean(-1) ** 0.5)
    reps = bootstrap_orders(y, H, 2.0, n_bootstrap=400, seed=11)
    jack = jackknife_orders(y, H, 2.0)
    np.testing.assert_allclose(
        bootstrap_interval(theta, reps, method="percentile"),
        [0.4725242753613592, 0.5546391174366104],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        bootstrap_interval(theta, reps, jack, method="bca"),
        [0.47040858655913814, 0.5464684389436927],
        rtol=1e-12,
    )


def test_bca_corrects_for_bias():
    # All replicates lie above the estimate, so the bootstrap distribution is
    # biased upwards: z0 is negative (clipped, not infinite) and the BCa
    # interval moves down relative to the percentile interval.
    reps = np.linspace(1.0, 2.0, 101)
    jack = np.array([-1.0, 1.0])
    lower, upper = bootstrap_interval(0.0, reps, jack, method="bca")
    p_lower, p_upper = bootstrap_interval(0.0, reps)
    assert lower == pytest.approx(reps[0])
    assert np.isfinite(upper)
    assert upper < p_upper
    assert lower <= p_lower


def test_bca_acceleration_follows_the_jackknife_formula():
    reps = np.linspace(-1.0, 1.0, 2001)
    jack = np.array([0.0, 0.0, 0.0, 3.0])  # skewed jackknife values
    d = jack.mean() - jack
    a = np.sum(d**3) / (6 * np.sum(d**2) ** 1.5)
    normal = NormalDist()
    z0 = 0.0  # the estimate is the median of the replicates
    levels = [
        normal.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        for z in (normal.inv_cdf(0.025), normal.inv_cdf(0.975))
    ]
    np.testing.assert_allclose(
        bootstrap_interval(0.0, reps, jack, method="bca"),
        np.quantile(reps, levels),
    )


def test_bca_with_extreme_acceleration():
    # A jackknife acceleration is at most 1/6 in magnitude; one outlier among
    # many jackknife values comes close to that bound (a = 0.166). At a very
    # high confidence level, z_{1-alpha/2} = 6.1, so 1 - a z < 0 for the upper
    # level. Taken literally, the formula would then put the upper level near
    # 0, below the lower level; the limit of the adjusted level is 1 instead.
    reps = np.linspace(-1.0, 1.0, 10001)  # median 0: z0 = 0
    jack = np.array([0.0] * 999 + [-1000.0])
    lower, upper = bootstrap_interval(
        0.0, reps, jack, method="bca", confidence=1 - 1e-9
    )
    assert upper == 1.0
    assert -1.0 < lower < 0.0


def test_constant_jackknife_means_no_acceleration():
    reps = np.linspace(-1.0, 1.0, 1001)
    np.testing.assert_allclose(
        bootstrap_interval(0.0, reps, np.ones(5), method="bca"),
        bootstrap_interval(0.0, reps),
    )


def test_interval_with_leading_axes_and_nan():
    reps = np.stack([np.linspace(0.0, 1.0, 11), np.full(11, np.nan)])
    lower, upper = bootstrap_interval(np.array([0.5, 0.5]), reps)
    assert lower[0] == pytest.approx(0.025)
    assert np.isnan(lower[1])
    assert np.isnan(upper[1])
    lower, upper = bootstrap_interval(np.array([np.nan]), reps[:1])
    assert np.isnan(lower[0])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "basic"}, "method must be one of"),
        ({"method": "bca"}, "needs the jackknife"),
        ({"method": "bca", "jackknife": np.ones(1)}, "jackknife must have shape"),
        ({"replicates": np.ones((2, 3))}, "replicates must have shape"),
    ],
)
def test_interval_rejects_bad_input(kwargs, match):
    args = {"estimate": 0.0, "replicates": np.ones(10)}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        bootstrap_interval(**args)
