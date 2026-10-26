"""The data-ownership and determinism contract of BrownianPath.

Several tests reproduce the counterexamples of the external architecture
audit of October 2026 (findings A1 to A3) and guard against their return.
"""

import numpy as np
import pytest

from numsde import BrownianPath, WienerProcess
from numsde.stepping import uniform_grid


def sample(seed=12, n_steps=4, n_paths=3, dim=2):
    return WienerProcess(dim).sample_path(
        uniform_grid(1.0, n_steps), n_paths=n_paths, seed=seed
    )


# -------------------------------------------- ownership and immutability (A1)


def test_audit_a1_mutating_a_coarse_path_is_impossible():
    path = WienerProcess().sample_path([0.0, 0.5, 1.0], seed=12)
    coarse = path.coarsen(2)
    with pytest.raises(ValueError, match="read-only"):
        coarse.values[-1, 0, 0] = 321.0
    assert path.values[-1, 0, 0] != 321.0


def test_values_are_read_only():
    path = sample()
    with pytest.raises(ValueError, match="read-only"):
        path.values[1, 0, 0] = 0.0


def test_views_of_values_cannot_be_made_writeable():
    view = sample().values[1:]
    with pytest.raises(ValueError, match="cannot set WRITEABLE"):
        view.flags.writeable = True


def test_constructor_copies_its_input():
    source = np.array([[[0.0]], [[0.5]]])
    path = BrownianPath([0.0, 1.0], source, seed=1)
    source[1, 0, 0] = 99.0
    assert path.values[1, 0, 0] == 0.5


@pytest.mark.parametrize(
    "derive",
    [
        lambda p: p.coarsen(2),
        lambda p: p.refine(2),
        lambda p: p.insert([0.1, 3.0]),
    ],
    ids=["coarsen", "refine", "insert"],
)
def test_derived_paths_share_no_memory(derive):
    path = sample()
    assert not np.shares_memory(derive(path).values, path.values)


def test_increments_are_a_new_array():
    path = sample()
    increments = path.increments
    increments[0, 0, 0] = 99.0
    assert path.increments[0, 0, 0] != 99.0


# ------------------------------------------------------- determinism (A2)


def test_audit_a2_refining_twice_gives_the_same_path():
    path = WienerProcess().sample_path([0.0, 1.0], seed=12)
    np.testing.assert_array_equal(path.refine(2).values, path.refine(2).values)


def test_insert_is_a_pure_function():
    path = sample()
    before = path.values.copy()
    a = path.insert([0.3, 0.6, 2.0])
    b = path.insert([2.0, 0.6, 0.3])  # order of the request does not matter
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(path.values, before)


def test_operations_do_not_depend_on_call_history():
    fresh = sample().refine(3)
    used = sample()
    used.refine(2)
    used.insert([0.42])
    used.coarsen(2)
    np.testing.assert_array_equal(used.refine(3).values, fresh.values)


def test_different_requests_use_different_random_numbers():
    path = WienerProcess().sample_path([0.0, 1.0], n_paths=4, seed=5)
    at_half = path.insert([0.5]).values[1]
    at_half_with_more = path.insert([0.5, 0.75]).values[1]
    assert not np.array_equal(at_half, at_half_with_more)


def test_unseeded_path_refines_reproducibly():
    path = WienerProcess().sample_path([0.0, 1.0], n_paths=2)
    np.testing.assert_array_equal(path.refine(2).values, path.refine(2).values)


def test_seed_sequence_property_returns_an_independent_copy():
    path = sample()
    expected = path.refine(2).values
    path.seed_sequence.spawn(5)  # would advance a shared SeedSequence
    np.testing.assert_array_equal(path.refine(2).values, expected)


def test_user_seed_sequence_is_not_shared():
    seed = np.random.SeedSequence(77)
    path = WienerProcess().sample_path([0.0, 1.0], seed=seed)
    expected = path.refine(2).values
    seed.spawn(3)
    np.testing.assert_array_equal(path.refine(2).values, expected)
    np.testing.assert_array_equal(
        WienerProcess().sample_path([0.0, 1.0], seed=77).values, path.values
    )


# ----------------------------------------------------- validation (A3)


@pytest.mark.parametrize("dtype", [np.int64, np.complex128, np.bool_])
def test_audit_a3_non_real_floating_values_are_rejected(dtype):
    values = np.zeros((2, 1, 1), dtype=dtype)
    with pytest.raises(TypeError, match="real floating-point"):
        BrownianPath([0.0, 1.0], values)


@pytest.mark.parametrize("shape", [(2, 0, 1), (2, 1, 0)])
def test_audit_a3_empty_batch_or_dimension_is_rejected(shape):
    with pytest.raises(ValueError, match="n_paths >= 1, dim >= 1"):
        BrownianPath([0.0, 1.0], np.zeros(shape))


def test_float32_values_are_kept():
    path = BrownianPath([0.0, 1.0], np.zeros((2, 1, 1), dtype=np.float32), seed=1)
    assert path.refine(2).values.dtype == np.float32
