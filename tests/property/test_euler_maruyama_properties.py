"""Invariants of the Euler–Maruyama scheme and the solver."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import numsde as ns
from numsde import ItoSDE, NoiseStructure, Solver, WienerProcess
from numsde.benchmarks import LinearSystem
from numsde.schemes import EulerMaruyama

seeds = st.integers(min_value=0, max_value=2**32 - 1)


@st.composite
def grids(draw, max_steps=8):
    steps = draw(
        st.lists(
            st.floats(min_value=1e-3, max_value=0.5), min_size=1, max_size=max_steps
        )
    )
    return np.concatenate([[0.0], np.cumsum(steps)])


class PureNoise(ItoSDE):
    """dX = dW in d = m dimensions."""

    structure = NoiseStructure(additive=True)

    def __init__(self, x0):
        super().__init__(dim=len(x0), noise_dim=len(x0), x0=list(x0))

    def drift(self, t, x):
        return 0.0 * x

    def diffusion(self, t, x):
        xp = ns.array_namespace(x)
        eye = ns.asarray_like(np.eye(self.dim), x)
        return xp.broadcast_to(eye, (*x.shape[:-1], self.dim, self.dim))


@given(grid=grids(), n_paths=st.integers(1, 5), seed=seeds)
def test_paths_are_solved_independently(grid, n_paths, seed):
    """Solving a batch equals solving each path on its own."""
    problem = LinearSystem()
    path = WienerProcess(2).sample_path(grid, n_paths=n_paths, seed=seed)
    batch = Solver(problem, EulerMaruyama()).solve(path).x
    for i in range(n_paths):
        single = ns.BrownianPath(path.times, path.values[:, i : i + 1], seed=0)
        alone = Solver(problem, EulerMaruyama()).solve(single).x
        np.testing.assert_allclose(batch[:, i : i + 1], alone, rtol=1e-14, atol=0)


@given(
    grid=grids(),
    a=st.lists(st.floats(-2.0, 2.0), min_size=2, max_size=2),
    seed=seeds,
)
def test_without_noise_the_scheme_is_explicit_euler(grid, a, seed):
    problem = LinearSystem(drift_eigenvalues=a, noise_eigenvalues=np.zeros((2, 2)))
    path = WienerProcess(2).sample_path(grid, n_paths=2, seed=seed)
    x = Solver(problem, EulerMaruyama()).solve(path).x
    expected = np.array(problem.x0)
    for n, h in enumerate(np.diff(grid)):
        expected = expected + h * problem.A @ expected
        np.testing.assert_allclose(x[n + 1, 0], expected, rtol=1e-12, atol=1e-14)


@given(
    grid=grids(),
    x0=st.lists(st.floats(-10.0, 10.0), min_size=1, max_size=3),
    seed=seeds,
)
def test_pure_noise_reproduces_the_brownian_path(grid, x0, seed):
    """For dX = dW the scheme is exact: X_h^n = x0 + W(t_n)."""
    problem = PureNoise(x0)
    path = WienerProcess(len(x0)).sample_path(grid, n_paths=3, seed=seed)
    x = Solver(problem, EulerMaruyama()).solve(path).x
    np.testing.assert_allclose(x, np.asarray(x0) + path.values, rtol=1e-12, atol=1e-12)


@given(n_steps=st.integers(1, 6), factor=st.integers(1, 4), seed=seeds)
def test_coarsened_paths_drive_coarse_solutions(n_steps, factor, seed):
    """A coarsened path is the same realization: the step-size-h solution on
    it equals a solution on a path built from the summed increments."""
    fine = WienerProcess(2).sample_path(
        ns.uniform_grid(1.0, n_steps * factor), n_paths=2, seed=seed
    )
    coarse = fine.coarsen(factor)
    rebuilt = ns.BrownianPath(
        coarse.times,
        np.concatenate(
            [
                np.zeros((1, 2, 2)),
                np.cumsum(
                    fine.increments.reshape(n_steps, factor, 2, 2).sum(axis=1), axis=0
                ),
            ]
        ),
        seed=0,
    )
    solver = Solver(LinearSystem(), EulerMaruyama())
    np.testing.assert_allclose(
        solver.solve(coarse).x, solver.solve(rebuilt).x, rtol=1e-12, atol=1e-12
    )
