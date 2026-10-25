"""Statistical checks of the sampled Brownian motion and Brownian bridge.

The tests use fixed seeds, so they are deterministic. Tolerances are a few
standard errors of the estimated moments, so a correct implementation passes
with overwhelming probability for any seed.
"""

import numpy as np
import pytest

from numsde import WienerProcess
from numsde.stepping import uniform_grid

pytestmark = pytest.mark.slow

N_PATHS = 400_000


def standard_error_of_variance(variance, n):
    # Var of the sample variance of a normal sample: 2 sigma^4 / (n - 1).
    return np.sqrt(2.0 / (n - 1)) * variance


def check_independent_gaussian_increments(path):
    """Mean 0, variance h_n, uncorrelated across steps and components."""
    dW = path.increments
    h = path.step_sizes[:, None]
    n = path.n_paths
    mean = dW.mean(axis=1)
    assert np.all(np.abs(mean) < 5 * np.sqrt(h / n))
    var = dW.var(axis=1)
    assert np.all(np.abs(var - h) < 5 * standard_error_of_variance(h, n))
    # Normalised increments must be uncorrelated across steps and components.
    z = (dW / np.sqrt(h)[:, :, None]).transpose(0, 2, 1).reshape(-1, n)
    corr = np.corrcoef(z)
    off_diagonal = corr[~np.eye(corr.shape[0], dtype=bool)]
    assert np.max(np.abs(off_diagonal)) < 5 / np.sqrt(n)


def test_sampled_increments():
    grid = np.array([0.0, 0.1, 0.15, 0.5, 1.0, 2.0])
    path = WienerProcess(2).sample_path(grid, n_paths=N_PATHS, seed=2026)
    check_independent_gaussian_increments(path)


def test_refined_increments_follow_the_bridge():
    coarse = WienerProcess(2).sample_path(uniform_grid(1.0, 2), n_paths=N_PATHS, seed=7)
    fine = coarse.refine(3)
    check_independent_gaussian_increments(fine)


def test_irregular_insertion():
    path = WienerProcess().sample_path([0.0, 1.0], n_paths=N_PATHS, seed=11)
    merged = path.insert([0.05, 0.3, 0.31, 0.9, 1.7, 2.5])
    check_independent_gaussian_increments(merged)


def test_bridge_variance_given_endpoints():
    # Conditional on W(0) = 0 and W(1), the midpoint has mean W(1)/2 and
    # variance 1/4; the residual must be independent of W(1).
    path = WienerProcess().sample_path([0.0, 1.0], n_paths=N_PATHS, seed=3)
    mid = path.insert([0.5]).values[1, :, 0]
    residual = mid - 0.5 * path.values[1, :, 0]
    assert abs(residual.var() - 0.25) < 5 * standard_error_of_variance(0.25, N_PATHS)
    assert abs(np.corrcoef(residual, path.values[1, :, 0])[0, 1]) < 5 / np.sqrt(N_PATHS)
