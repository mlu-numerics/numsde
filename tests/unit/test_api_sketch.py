"""The API sketch of the architecture document, as a runnable test.

The convergence-study part of the sketch follows with the analysis module.
"""

import numpy as np

import numsde as ns


class GeometricBrownianMotion(ns.ItoSDE):
    """dX = mu X dt + sigma X dW, with exact solution."""

    structure = ns.NoiseStructure(scalar=True)

    def __init__(self, mu=0.05, sigma=0.2, x0=1.0):
        super().__init__(dim=1, noise_dim=1, x0=x0)
        self.mu, self.sigma = mu, sigma

    def drift(self, t, x):
        return self.mu * x

    def diffusion(self, t, x):
        return (self.sigma * x)[..., None]  # shape (..., d, m)

    def exact_solution(self, t, W):
        xp = ns.array_namespace(W)
        return self.x0 * xp.exp((self.mu - 0.5 * self.sigma**2) * t + self.sigma * W)


def test_api_sketch_end_to_end():
    problem = GeometricBrownianMotion()
    path = ns.WienerProcess(1).sample_path(
        ns.uniform_grid(1.0, 2**10), n_paths=5_000, seed=1
    )

    solution = ns.Solver(problem, ns.schemes.EulerMaruyama()).solve(path.coarsen(4))

    assert solution.x.shape == (2**8 + 1, 5_000, 1)
    assert problem.has_exact_solution
    exact = problem.exact_solution(1.0, path.values[-1])
    rms_error = np.sqrt(np.mean((solution.final - exact) ** 2))
    assert rms_error < 0.01  # the EM error for h = 2^-8 is of order 1e-3


def test_user_defined_problem_matches_the_benchmark():
    path = ns.WienerProcess(1).sample_path(ns.uniform_grid(1.0, 16), n_paths=3, seed=2)
    scheme = ns.schemes.EulerMaruyama()
    user = ns.Solver(GeometricBrownianMotion(), scheme).solve(path)
    library = ns.Solver(ns.benchmarks.GeometricBrownianMotion(), scheme).solve(path)
    np.testing.assert_array_equal(user.x, library.x)
