"""Backend consistency: the same scheme code runs on PyTorch."""

import numpy as np
import pytest

from numsde import Solver, WienerProcess
from numsde.benchmarks import GeometricBrownianMotion, LinearSystem
from numsde.schemes import EulerMaruyama
from numsde.stepping import uniform_grid

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "problem", [GeometricBrownianMotion(), LinearSystem()], ids=["gbm", "linear"]
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
