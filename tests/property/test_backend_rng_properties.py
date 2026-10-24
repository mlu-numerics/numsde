"""Property-based tests for random number generation."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from numsde import RandomNumberGenerator

seeds = st.integers(min_value=0, max_value=2**63 - 1)
shapes = st.lists(st.integers(min_value=0, max_value=6), min_size=0, max_size=3).map(
    tuple
)


@given(seed=seeds, shape=shapes)
def test_shape_and_reproducibility(seed, shape):
    a = RandomNumberGenerator(seed).standard_normal(shape)
    b = RandomNumberGenerator(seed).standard_normal(shape)
    assert a.shape == shape
    np.testing.assert_array_equal(a, b)
    assert np.all(np.isfinite(a))


@given(seed=seeds, n=st.integers(min_value=1, max_value=8))
def test_spawned_children_are_reproducible(seed, n):
    a = [r.standard_normal(4) for r in RandomNumberGenerator(seed).spawn(n)]
    b = [r.standard_normal(4) for r in RandomNumberGenerator(seed).spawn(n)]
    assert len(a) == n
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


@given(seed=seeds)
def test_spawn_does_not_change_parent_stream(seed):
    parent = RandomNumberGenerator(seed)
    parent.spawn(2)
    np.testing.assert_array_equal(
        parent.standard_normal(5), RandomNumberGenerator(seed).standard_normal(5)
    )
