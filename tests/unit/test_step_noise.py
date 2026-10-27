import numpy as np
import pytest

from numsde import BrownianPath, NoiseSource, StepNoise, WienerProcess
from numsde.noise import STEP_QUANTITIES
from numsde.stepping import uniform_grid


def test_step_noise_defaults_and_immutability():
    noise = StepNoise(t=0.0, h=0.5, dW=np.zeros((2, 1)))
    assert noise.I is None
    assert noise.Z is None
    with pytest.raises(AttributeError):
        noise.h = 1.0


def test_brownian_path_is_a_noise_source():
    path = WienerProcess(2).sample_path(uniform_grid(1.0, 4), n_paths=3, seed=1)
    assert isinstance(path, NoiseSource)
    assert path.provides == {"dW"}
    assert path.namespace.__name__ == "array_api_compat.numpy"
    assert path.dtype == np.float64
    assert path.device == "cpu"
    assert path.n_steps == 4


def test_steps_yield_increments_in_time_order():
    grid = np.array([0.0, 0.1, 0.5, 1.0])
    path = WienerProcess(2).sample_path(grid, n_paths=3, seed=4)
    steps = list(path.steps())
    assert [s.t for s in steps] == [0.0, 0.1, 0.5]
    np.testing.assert_allclose([s.h for s in steps], np.diff(grid))
    np.testing.assert_array_equal(np.stack([s.dW for s in steps]), path.increments)
    assert all(isinstance(s.t, float) and isinstance(s.h, float) for s in steps)


def test_steps_are_reproducible_and_do_not_alias_the_path():
    path = WienerProcess().sample_path(uniform_grid(1.0, 2), n_paths=2, seed=2)
    first = [s.dW for s in path.steps()]
    first[0][...] = 99.0  # increments are new arrays
    second = [s.dW for s in path.steps()]
    np.testing.assert_array_equal(second[0], path.increments[0])


def test_steps_of_a_single_point_path_are_empty():
    path = BrownianPath([0.0], np.zeros((1, 1, 1)), seed=0)
    assert list(path.steps()) == []


def test_unknown_step_quantity_is_rejected():
    path = WienerProcess().sample_path([0.0, 1.0], seed=0)
    with pytest.raises(ValueError, match="Unknown step quantities"):
        list(path.steps(["dV"]))


@pytest.mark.parametrize("quantity", ["I", "Z"])
def test_higher_quantities_are_not_provided_yet(quantity):
    path = WienerProcess().sample_path([0.0, 1.0], seed=0)
    assert quantity in STEP_QUANTITIES
    with pytest.raises(ValueError, match="cannot provide"):
        list(path.steps({"dW", quantity}))
