"""Empirical strong and weak convergence of the Euler–Maruyama scheme.

All step sizes are driven by one Brownian realization, sampled on the finest
grid and coarsened. The strong error is the default error of the library,
uniformly on the grid, with p = 2. Seeds are fixed, so the tests are
deterministic; the tolerances leave room for the Monte Carlo error of the
chosen sample sizes. The convergence-study module with bootstrap intervals
for the order follows later in milestone M1.
"""

import numpy as np
import pytest

from numsde import Solver, WienerProcess
from numsde.benchmarks import GeometricBrownianMotion, LinearSystem
from numsde.schemes import EulerMaruyama
from numsde.stepping import uniform_grid

pytestmark = pytest.mark.slow

FINEST = 2**12
FACTORS = [2**k for k in range(3, 9)]  # step sizes 2^-9 ... 2^-4


def strong_errors(problem, n_paths, seed):
    path = WienerProcess(problem.noise_dim).sample_path(
        uniform_grid(1.0, FINEST), n_paths=n_paths, seed=seed
    )
    solver = Solver(problem, EulerMaruyama())
    steps, errors = [], []
    for factor in FACTORS:
        coarse = path.coarsen(factor)
        x = solver.solve(coarse).x
        exact = problem.exact_solution(coarse.times[:, None, None], coarse.values)
        pathwise = np.max(np.linalg.norm(x - exact, axis=-1), axis=0)
        steps.append(coarse.step_sizes[0])
        errors.append(np.sqrt(np.mean(pathwise**2)))
    return np.array(steps), np.array(errors)


def fitted_order(steps, errors):
    return np.polyfit(np.log(steps), np.log(errors), 1)[0]


@pytest.mark.parametrize(
    "problem",
    [GeometricBrownianMotion(mu=1.0, sigma=1.0), LinearSystem()],
    ids=["gbm", "linear_system"],
)
def test_strong_order_one_half(problem):
    steps, errors = strong_errors(problem, n_paths=10_000, seed=2026)
    expected = EulerMaruyama().expected_strong_order(problem.structure)
    assert expected == 0.5
    assert abs(fitted_order(steps, errors) - expected) < 0.1
    assert np.all(np.diff(errors) > 0)  # larger steps, larger errors


def test_weak_order_one_for_the_mean_of_gbm():
    # E[X(1)] = x0 exp(mu); Euler-Maruyama gives E[X_h^N] = x0 (1 + mu h)^N
    # exactly, so the weak error of phi(x) = x is known without Monte Carlo
    # bias. The Monte Carlo estimate must agree within five standard errors.
    problem = GeometricBrownianMotion(mu=1.0, sigma=0.5)
    n_paths, n_steps = 400_000, 8
    path = WienerProcess(1).sample_path(
        uniform_grid(1.0, n_steps), n_paths=n_paths, seed=7
    )
    final = Solver(problem, EulerMaruyama()).solve(path, save="final").final[:, 0]
    h = 1.0 / n_steps
    exact_mean_scheme = (1 + h) ** n_steps
    standard_error = final.std(ddof=1) / np.sqrt(n_paths)
    assert abs(final.mean() - exact_mean_scheme) < 5 * standard_error
    # The bias of the scheme decays like h: e^1 - (1 + h)^(1/h) ~ e h / 2.
    bias = np.e - exact_mean_scheme
    assert 0.4 * np.e * h < bias < 0.5 * np.e * h
