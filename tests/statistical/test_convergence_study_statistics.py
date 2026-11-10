"""Statistical properties of the convergence study and its bootstrap intervals.

The coverage test uses synthetic per-path data whose population order is
known exactly: Y_ik = h_k^(p q) Z_ik with E[Z_ik] = 1, correlated across step
sizes like the errors of one path. It checks that the 95 % intervals cover
the true order in about 95 % of repeated studies.
"""

import numpy as np
import pytest

from numsde import WienerProcess
from numsde.analysis import StrongConvergenceStudy
from numsde.analysis.statistics import (
    bootstrap_interval,
    bootstrap_orders,
    fit_order,
    jackknife_orders,
)
from numsde.benchmarks import GeometricBrownianMotion, LinearSystem
from numsde.schemes import EulerMaruyama
from numsde.stepping import uniform_grid

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("method", ["percentile", "bca"])
def test_bootstrap_intervals_cover_the_true_order(method):
    p, q, rho, scale = 2.0, 0.5, 0.8, 0.5
    h = 2.0 ** -np.arange(3, 8)
    n_paths, n_studies = 500, 1000
    rng = np.random.default_rng(12345)
    hits = 0
    for r in range(n_studies):
        common = rng.normal(size=n_paths)
        own = rng.normal(size=(h.size, n_paths))
        z = np.exp(scale * (rho * common + np.sqrt(1 - rho**2) * own) - scale**2 / 2)
        y = h[:, None] ** (p * q) * z
        theta = fit_order(h, y.mean(-1) ** (1 / p))
        reps = bootstrap_orders(y, h, p, n_bootstrap=500, seed=r)
        jack = jackknife_orders(y, h, p) if method == "bca" else None
        lower, upper = bootstrap_interval(theta, reps, jack, method=method)
        hits += lower <= q <= upper
    # Standard error of the coverage estimate: about 0.007.
    assert 0.92 <= hits / n_studies <= 0.975


@pytest.mark.parametrize(
    "problem",
    [GeometricBrownianMotion(mu=1.0, sigma=1.0), LinearSystem()],
    ids=["gbm", "linear_system"],
)
def test_study_finds_order_one_half(problem):
    path = WienerProcess(problem.noise_dim).sample_path(
        uniform_grid(1.0, 2**12), n_paths=10_000, seed=2026
    )
    factors = [2**k for k in range(3, 9)]
    result = StrongConvergenceStudy(problem, EulerMaruyama(), path, factors).run(seed=1)
    order = float(result.orders[0])
    lower, upper = result.orders_ci[0]
    assert abs(order - 0.5) < 0.1
    assert lower < order < upper
    assert upper - lower < 0.2
