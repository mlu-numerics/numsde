from typing import ClassVar

import numpy as np
import pytest

import numsde as ns
from numsde import BrownianPath, ItoSDE, NoiseSource, Solution, Solver, WienerProcess
from numsde.benchmarks import GeometricBrownianMotion, LinearSystem
from numsde.schemes import EulerMaruyama, OneStepScheme
from numsde.stepping import uniform_grid


def gbm_path(n_steps=8, n_paths=3, seed=1, **kwargs):
    return WienerProcess(1).sample_path(
        uniform_grid(1.0, n_steps), n_paths=n_paths, seed=seed, **kwargs
    )


def test_solve_with_a_path():
    solution = Solver(GeometricBrownianMotion(), EulerMaruyama()).solve(gbm_path())
    assert isinstance(solution, Solution)
    assert solution.x.shape == (9, 3, 1)
    np.testing.assert_array_equal(solution.t, uniform_grid(1.0, 8))
    np.testing.assert_array_equal(solution.x[0], 1.0)
    assert not solution.x.flags.writeable
    assert not solution.t.flags.writeable


def test_solution_matches_a_manual_loop():
    problem = LinearSystem()
    path = WienerProcess(2).sample_path(uniform_grid(1.0, 5), n_paths=4, seed=3)
    solution = Solver(problem, EulerMaruyama()).solve(path)
    x = np.broadcast_to(problem.x0, (4, 2)).copy()
    for n, h in enumerate(path.step_sizes):
        dW = path.increments[n]
        x = x + x @ problem.A.T * h + np.einsum("jkl,nl,nj->nk", problem.B, x, dW)
        np.testing.assert_allclose(solution.x[n + 1], x, rtol=1e-13)


def test_solution_properties_and_metadata():
    problem = GeometricBrownianMotion()
    scheme = EulerMaruyama()
    solution = Solver(problem, scheme).solve(gbm_path())
    assert solution.problem is problem
    assert solution.scheme is scheme
    assert (solution.n_paths, solution.dim) == (3, 1)
    np.testing.assert_array_equal(solution.final, solution.x[-1])
    assert dict(solution.metadata) == {
        "scheme": "Euler–Maruyama",
        "n_steps": 8,
        "n_paths": 3,
        "backend": "array_api_compat.numpy",
        "dtype": "float64",
        "device": "cpu",
        "save": "all",
        "structure": frozenset({"scalar", "commutative"}),
        "verified": frozenset({"scalar", "commutative"}),
    }
    with pytest.raises(TypeError):
        solution.metadata["scheme"] = "other"
    assert repr(solution) == (
        "Solution(scheme='Euler–Maruyama', n_times=9, n_paths=3, dim=1)"
    )


def test_save_final_keeps_only_the_last_value():
    solver = Solver(GeometricBrownianMotion(), EulerMaruyama())
    path = gbm_path()
    full, final = solver.solve(path), solver.solve(path, save="final")
    assert final.x.shape == (1, 3, 1)
    np.testing.assert_array_equal(final.t, [1.0])
    np.testing.assert_array_equal(final.x[0], full.x[-1])
    assert final.metadata["save"] == "final"


def test_rejects_unknown_save_mode():
    solver = Solver(GeometricBrownianMotion(), EulerMaruyama())
    with pytest.raises(ValueError, match="save must be"):
        solver.solve(gbm_path(), save="some")


def test_same_path_gives_the_same_solution():
    solver = Solver(GeometricBrownianMotion(), EulerMaruyama())
    path = gbm_path()
    np.testing.assert_array_equal(solver.solve(path).x, solver.solve(path).x)


# ------------------------------------------------------------- sampling


def test_solver_samples_a_path_from_times_and_seed():
    solver = Solver(GeometricBrownianMotion(), EulerMaruyama())
    grid = uniform_grid(1.0, 8)
    sampled = solver.solve(times=grid, n_paths=3, seed=1)
    np.testing.assert_array_equal(sampled.x, solver.solve(gbm_path()).x)


def test_sampling_defaults_to_one_path():
    solver = Solver(GeometricBrownianMotion(), EulerMaruyama())
    assert solver.solve(times=[0.0, 1.0], seed=0).n_paths == 1


@pytest.mark.parametrize("backend", ["numpy", ns.array_namespace()])
def test_sampling_with_explicit_backend_and_dtype(backend):
    solver = Solver(GeometricBrownianMotion(), EulerMaruyama())
    solution = solver.solve(
        times=[0.0, 0.5, 1.0], seed=0, backend=backend, dtype=np.float32
    )
    assert solution.x.dtype == np.float32


def test_sampling_follows_an_array_initial_value():
    problem = LinearSystem(x0=np.array([1.0, 0.5], dtype=np.float32))
    solution = Solver(problem, EulerMaruyama()).solve(times=[0.0, 1.0], seed=0)
    assert solution.x.dtype == np.float32
    explicit = Solver(problem, EulerMaruyama()).solve(
        times=[0.0, 1.0], seed=0, dtype=np.float64, device="cpu"
    )
    assert explicit.x.dtype == np.float64


def test_needs_noise_or_times():
    with pytest.raises(ValueError, match="noise source or the times"):
        Solver(GeometricBrownianMotion(), EulerMaruyama()).solve()


@pytest.mark.parametrize(
    "option",
    [
        {"times": [0.0, 1.0]},
        {"n_paths": 2},
        {"seed": 1},
        {"backend": "numpy"},
        {"dtype": np.float32},
        {"device": "cpu"},
    ],
)
def test_sampling_options_cannot_be_combined_with_noise(option):
    solver = Solver(GeometricBrownianMotion(), EulerMaruyama())
    with pytest.raises(TypeError, match="cannot be combined"):
        solver.solve(gbm_path(), **option)


# ---------------------------------------------------------------- checks


def test_rejects_wrong_argument_types():
    with pytest.raises(TypeError, match="problem must be an instance of SDEProblem"):
        Solver("gbm", EulerMaruyama())
    with pytest.raises(TypeError, match="scheme must be an instance of Scheme"):
        Solver(GeometricBrownianMotion(), EulerMaruyama)
    with pytest.raises(TypeError, match="noise must be a NoiseSource"):
        Solver(GeometricBrownianMotion(), EulerMaruyama()).solve(np.zeros((2, 1, 1)))


def test_solver_checks_the_scheme_against_the_problem():
    class NeedsAdditive(OneStepScheme):
        name: ClassVar[str] = "needs additive"
        strong_order: ClassVar[float] = 1.0
        requires_structure: ClassVar[frozenset[str]] = frozenset({"additive"})

        def step(self, problem, t, x, noise, state):
            return x  # pragma: no cover - never called

    with pytest.raises(ValueError, match="requires noise structure"):
        Solver(GeometricBrownianMotion(), NeedsAdditive())


def test_rejects_noise_of_wrong_dimension():
    path = WienerProcess(2).sample_path([0.0, 1.0], seed=0)
    with pytest.raises(ValueError, match="noise_dim=1"):
        Solver(GeometricBrownianMotion(), EulerMaruyama()).solve(path)


def test_rejects_noise_starting_at_another_time():
    path = WienerProcess(1).sample_path([0.5, 1.0], seed=0)
    with pytest.raises(ValueError, match="initial time"):
        Solver(GeometricBrownianMotion(), EulerMaruyama()).solve(path)


def test_noise_grid_may_start_at_a_nonzero_initial_time():
    path = WienerProcess(1).sample_path([0.5, 1.0], seed=0)
    solution = Solver(GeometricBrownianMotion(t0=0.5), EulerMaruyama()).solve(path)
    assert solution.t[0] == 0.5


def test_rejects_noise_with_another_dtype_than_x0():
    problem = LinearSystem(x0=np.array([1.0, 1.0], dtype=np.float32))
    path = WienerProcess(2).sample_path([0.0, 1.0], seed=0)
    with pytest.raises(ValueError, match="differ in dtype or device"):
        Solver(problem, EulerMaruyama()).solve(path)


def test_array_initial_value_matching_the_noise():
    problem = LinearSystem(x0=np.array([1.0, 1.0], dtype=np.float32))
    path = WienerProcess(2).sample_path([0.0, 1.0], seed=0, dtype=np.float32)
    assert Solver(problem, EulerMaruyama()).solve(path).x.dtype == np.float32


class ForeignNamespacePath(NoiseSource):
    """A path that reports a different array library."""

    def __init__(self, path):
        self._path = path

    times = property(lambda self: self._path.times)
    n_paths = property(lambda self: self._path.n_paths)
    dim = property(lambda self: self._path.dim)
    namespace = property(lambda self: np)  # plain NumPy, not array_api_compat
    dtype = property(lambda self: self._path.dtype)
    device = property(lambda self: self._path.device)

    def _steps(self, requires):
        return self._path._steps(requires)  # pragma: no cover - never called


def test_rejects_noise_from_another_array_library_than_x0():
    problem = LinearSystem(x0=np.array([1.0, 1.0]))
    path = ForeignNamespacePath(WienerProcess(2).sample_path([0.0, 1.0], seed=0))
    assert path.n_steps == 1
    with pytest.raises(ValueError, match="different array libraries"):
        Solver(problem, EulerMaruyama()).solve(path)


class BadShapes(ItoSDE):
    def __init__(self, drift_shape, diffusion_shape):
        super().__init__(dim=2, noise_dim=1, x0=[1.0, 1.0])
        self.drift_shape, self.diffusion_shape = drift_shape, diffusion_shape

    def drift(self, t, x):
        return np.zeros(self.drift_shape) if self.drift_shape is not None else 0.0

    def diffusion(self, t, x):
        return np.zeros(self.diffusion_shape)


@pytest.mark.parametrize(
    ("drift_shape", "diffusion_shape", "match"),
    [
        ((3, 1), (3, 2, 1), r"BadShapes.drift must return .*\(3, 2\).*got \(3, 1\)"),
        (None, (3, 2, 1), r"BadShapes.drift .* got \(\)"),
        ((3, 2), (3, 2), r"BadShapes.diffusion must return .*\(3, 2, 1\)"),
    ],
)
def test_rejects_coefficients_of_wrong_shape(drift_shape, diffusion_shape, match):
    path = WienerProcess(1).sample_path([0.0, 1.0], n_paths=3, seed=0)
    with pytest.raises(ValueError, match=match):
        Solver(BadShapes(drift_shape, diffusion_shape), EulerMaruyama()).solve(path)


def test_rejects_scheme_requirements_the_source_cannot_meet():
    class NeedsLevyAreas(OneStepScheme):
        name: ClassVar[str] = "needs I"
        strong_order: ClassVar[float] = 1.0
        requires: ClassVar[frozenset[str]] = frozenset({"dW", "I"})

        def step(self, problem, t, x, noise, state):
            return x  # pragma: no cover - never called

    solver = Solver(GeometricBrownianMotion(), NeedsLevyAreas())
    with pytest.raises(ValueError, match="cannot provide"):
        solver.solve(gbm_path())


def test_solver_properties():
    problem, scheme = GeometricBrownianMotion(), EulerMaruyama()
    solver = Solver(problem, scheme)
    assert solver.problem is problem
    assert solver.scheme is scheme


def test_wrapped_reference_path():
    # A path built from given values, e.g. reference data of a legacy script.
    values = np.array([[[0.0]], [[0.2]], [[-0.1]]])
    path = BrownianPath([0.0, 0.5, 1.0], values, seed=0)
    solution = Solver(GeometricBrownianMotion(mu=0.0, sigma=1.0), EulerMaruyama())
    x = solution.solve(path).x[:, 0, 0]
    np.testing.assert_allclose(x, [1.0, 1.2, 1.2 * (1 - 0.3)])
