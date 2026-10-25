import numpy as np
import pytest

from numsde.stepping import as_time_grid, uniform_grid


def test_uniform_grid_endpoints_and_spacing():
    grid = uniform_grid(2.0, 8, t0=1.0)
    assert grid[0] == 1.0
    assert grid[-1] == 2.0
    np.testing.assert_allclose(np.diff(grid), 0.125)


@pytest.mark.parametrize("n_steps", [0, -3, 2.5])
def test_uniform_grid_rejects_bad_step_count(n_steps):
    with pytest.raises(ValueError, match="n_steps"):
        uniform_grid(1.0, n_steps)


def test_uniform_grid_rejects_empty_interval():
    with pytest.raises(ValueError, match="t_end must be larger"):
        uniform_grid(1.0, 4, t0=1.0)


def test_as_time_grid_returns_read_only_copy():
    source = [0.0, 0.5, 1.0]
    grid = as_time_grid(source)
    assert grid.dtype == np.float64
    assert not grid.flags.writeable


@pytest.mark.parametrize(
    ("times", "message"),
    [
        ([[0.0, 1.0]], "one-dimensional"),
        ([0.0], "at least 2"),
        ([0.0, np.nan], "finite"),
        ([0.0, 1.0, 1.0], "strictly increasing"),
        ([1.0, 0.0], "strictly increasing"),
    ],
)
def test_as_time_grid_rejects_invalid_grids(times, message):
    with pytest.raises(ValueError, match=message):
        as_time_grid(times)


def test_as_time_grid_allows_single_point_on_request():
    assert as_time_grid([0.0], min_points=1).shape == (1,)
