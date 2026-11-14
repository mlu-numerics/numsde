"""Backend consistency: the same scheme code runs on PyTorch."""

import numpy as np
import pytest

from numsde import Solver, WienerProcess
from numsde.benchmarks import (
    BrownianMotionOnCircle,
    GeometricBrownianMotion,
    LinearSystem,
)
from numsde.schemes import EulerMaruyama
from numsde.stepping import uniform_grid

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "problem",
    [GeometricBrownianMotion(), LinearSystem(), BrownianMotionOnCircle()],
    ids=["gbm", "linear", "circle"],
)
def test_torch_solution_equals_numpy_solution(problem):
    grid = uniform_grid(1.0, 16)
    kwargs = {"n_paths": 4, "seed": 3}
    ref = WienerProcess(problem.noise_dim).sample_path(grid, **kwargs)
    path = WienerProcess(problem.noise_dim).sample_path(grid, backend="torch", **kwargs)
    solver = Solver(problem, EulerMaruyama())
    x = solver.solve(path).x
    assert isinstance(x, torch.Tensor)
    np.testing.assert_allclose(x.numpy(), solver.solve(ref).x, rtol=1e-13)


def test_torch_exact_solution_equals_numpy():
    problem = LinearSystem()
    W = np.array([[0.2, -0.4], [1.0, 0.3]])
    np.testing.assert_allclose(
        problem.exact_solution(0.7, torch.asarray(W)).numpy(),
        problem.exact_solution(0.7, W),
        rtol=1e-14,
    )


def test_tensor_initial_value_selects_the_backend():
    problem = LinearSystem(x0=torch.tensor([1.0, 0.5], dtype=torch.float32))
    solution = Solver(problem, EulerMaruyama()).solve(times=[0.0, 0.5, 1.0], seed=1)
    assert isinstance(solution.x, torch.Tensor)
    assert solution.x.dtype == torch.float32
    assert solution.metadata["backend"] == "array_api_compat.torch"


def test_sampling_on_the_torch_backend_by_name():
    solver = Solver(GeometricBrownianMotion(), EulerMaruyama())
    solution = solver.solve(times=[0.0, 1.0], seed=0, backend="torch")
    assert isinstance(solution.x, torch.Tensor)


def test_tensor_initial_value_rejects_numpy_noise():
    problem = LinearSystem(x0=torch.tensor([1.0, 0.5], dtype=torch.float64))
    path = WienerProcess(2).sample_path([0.0, 1.0], seed=0)
    with pytest.raises(ValueError, match="different array libraries"):
        Solver(problem, EulerMaruyama()).solve(path)


@pytest.mark.gpu
def test_cuda_solution_matches_numpy():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    problem = LinearSystem()
    grid = uniform_grid(1.0, 16)
    ref = WienerProcess(2).sample_path(grid, n_paths=8, seed=5)
    path = WienerProcess(2).sample_path(
        grid, n_paths=8, seed=5, backend="torch", device="cuda"
    )
    x = Solver(problem, EulerMaruyama()).solve(path).x
    assert x.device.type == "cuda"
    np.testing.assert_allclose(
        x.cpu().numpy(), Solver(problem, EulerMaruyama()).solve(ref).x, rtol=1e-12
    )


def test_sampling_rejects_numpy_for_a_tensor_initial_value():
    # Regression test for review finding P1 on PR #12.
    problem = LinearSystem(x0=torch.tensor([1.0, 0.5], dtype=torch.float64))
    solver = Solver(problem, EulerMaruyama())
    with pytest.raises(ValueError, match="different array libraries"):
        solver.solve(times=[0.0, 1.0], seed=0, backend="numpy")


def test_to_numpy_converts_tensors():
    from numsde._backend import to_numpy

    converted = to_numpy(torch.arange(3.0, dtype=torch.float64))
    assert isinstance(converted, np.ndarray)
    np.testing.assert_array_equal(converted, [0.0, 1.0, 2.0])


def test_convergence_study_on_torch_equals_numpy():
    from numsde.analysis import StrongConvergenceStudy

    problem = LinearSystem()
    grid = uniform_grid(1.0, 32)
    kwargs = {"n_paths": 40, "seed": 4}
    ref = WienerProcess(2).sample_path(grid, **kwargs)
    path = WienerProcess(2).sample_path(grid, backend="torch", **kwargs)
    for reference in ("exact", EulerMaruyama()):
        args = {"reference": reference, "reference_refinement": 1}
        on_numpy = StrongConvergenceStudy(problem, EulerMaruyama(), ref, [2, 4], **args)
        on_torch = StrongConvergenceStudy(
            problem, EulerMaruyama(), path, [2, 4], **args
        )
        np.testing.assert_allclose(on_torch.simulate(), on_numpy.simulate(), rtol=1e-12)


@pytest.mark.gpu
def test_convergence_study_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from numsde.analysis import StrongConvergenceStudy

    grid = uniform_grid(1.0, 64)
    ref = WienerProcess(1).sample_path(grid, n_paths=200, seed=9)
    path = WienerProcess(1).sample_path(
        grid, n_paths=200, seed=9, backend="torch", device="cuda"
    )
    problem = GeometricBrownianMotion(mu=1.0, sigma=1.0)
    on_cuda = StrongConvergenceStudy(problem, EulerMaruyama(), path, [1, 2, 4])
    on_numpy = StrongConvergenceStudy(problem, EulerMaruyama(), ref, [1, 2, 4])
    np.testing.assert_allclose(on_cuda.simulate(), on_numpy.simulate(), rtol=1e-12)


def test_torch_circle_exact_solution_equals_numpy():
    problem = BrownianMotionOnCircle(sigma=1.3, x0=(0.6, 0.8))
    W = np.array([[0.2], [-1.4]])
    np.testing.assert_allclose(
        problem.exact_solution(0.0, torch.asarray(W)).numpy(),
        problem.exact_solution(0.0, W),
        rtol=1e-14,
    )
