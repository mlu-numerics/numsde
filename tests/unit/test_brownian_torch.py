"""Backend consistency: PyTorch paths equal NumPy paths for the same seed."""

import numpy as np
import pytest

from numsde import WienerProcess
from numsde.stepping import uniform_grid

torch = pytest.importorskip("torch")


def test_torch_path_equals_numpy_path():
    grid = uniform_grid(1.0, 4)
    ref = WienerProcess(2).sample_path(grid, n_paths=3, seed=8)
    path = WienerProcess(2).sample_path(grid, n_paths=3, seed=8, backend="torch")
    assert isinstance(path.values, torch.Tensor)
    np.testing.assert_allclose(path.values.numpy(), ref.values)


def test_torch_refine_insert_coarsen_match_numpy():
    grid = uniform_grid(1.0, 2)
    ref = WienerProcess().sample_path(grid, n_paths=2, seed=4)
    path = WienerProcess().sample_path(grid, n_paths=2, seed=4, backend="torch")
    np.testing.assert_allclose(path.refine(3).values.numpy(), ref.refine(3).values)
    np.testing.assert_allclose(
        path.insert([0.2, 0.21, 1.5]).values.numpy(),
        ref.insert([0.2, 0.21, 1.5]).values,
    )
    np.testing.assert_allclose(
        path.coarsen(2).increments.numpy(), ref.coarsen(2).increments
    )


@pytest.mark.gpu
def test_cuda_path_matches_numpy():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    grid = uniform_grid(1.0, 4)
    ref = WienerProcess().sample_path(grid, n_paths=5, seed=1)
    path = WienerProcess().sample_path(
        grid, n_paths=5, seed=1, backend="torch", device="cuda"
    )
    assert path.values.device.type == "cuda"
    np.testing.assert_allclose(
        path.refine(2).values.cpu().numpy(), ref.refine(2).values
    )
