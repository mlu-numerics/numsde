"""Property-based tests for Brownian paths."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from numsde import WienerProcess

seeds = st.integers(min_value=0, max_value=2**32 - 1)
dims = st.integers(min_value=1, max_value=3)
n_paths = st.integers(min_value=1, max_value=4)


@st.composite
def grids(draw, min_steps=1, max_steps=12):
    steps = draw(
        st.lists(
            st.floats(min_value=1e-3, max_value=1.0),
            min_size=min_steps,
            max_size=max_steps,
        )
    )
    t0 = draw(st.floats(min_value=-1.0, max_value=1.0))
    return t0 + np.concatenate([[0.0], np.cumsum(steps)])


@given(grid=grids(), seed=seeds, dim=dims, n=n_paths)
def test_shape_and_zero_start(grid, seed, dim, n):
    path = WienerProcess(dim).sample_path(grid, n_paths=n, seed=seed)
    assert path.values.shape == (grid.size, n, dim)
    assert np.all(path.values[0] == 0.0)
    assert np.all(np.isfinite(path.values))


@given(grid=grids(), seed=seeds, factor=st.integers(min_value=1, max_value=5))
def test_refine_then_coarsen_reproduces_path(grid, seed, factor):
    path = WienerProcess(2).sample_path(grid, n_paths=2, seed=seed)
    np.testing.assert_array_equal(
        path.refine(factor).coarsen(factor).values, path.values
    )


@given(grid=grids(min_steps=2), seed=seeds, data=st.data())
def test_coarsening_sums_increments(grid, seed, data):
    path = WienerProcess(2).sample_path(grid, n_paths=3, seed=seed)
    divisors = [k for k in range(1, path.n_steps + 1) if path.n_steps % k == 0]
    factor = data.draw(st.sampled_from(divisors))
    coarse = path.coarsen(factor)
    blocks = path.increments.reshape(
        path.n_steps // factor, factor, path.n_paths, path.dim
    )
    np.testing.assert_allclose(coarse.increments, blocks.sum(axis=1), atol=1e-12)


@given(
    grid=grids(),
    seed=seeds,
    extra=st.lists(st.floats(min_value=0.0, max_value=5.0), max_size=8),
)
def test_insert_preserves_existing_values(grid, seed, extra):
    path = WienerProcess().sample_path(grid, n_paths=2, seed=seed)
    merged = path.insert(grid[0] + np.asarray(extra))
    keep = np.isin(merged.times, path.times)
    assert keep.sum() == grid.size
    np.testing.assert_array_equal(merged.values[keep], path.values)
    assert np.all(np.diff(merged.times) > 0)


@given(grid=grids(), seed=seeds)
def test_same_seed_same_path(grid, seed):
    a = WienerProcess(2).sample_path(grid, n_paths=2, seed=seed)
    b = WienerProcess(2).sample_path(grid, n_paths=2, seed=seed)
    np.testing.assert_array_equal(a.values, b.values)
