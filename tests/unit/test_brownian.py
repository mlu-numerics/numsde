import numpy as np
import pytest

from numsde import BrownianPath, NoiseProcess, RandomNumberGenerator, WienerProcess
from numsde.stepping import uniform_grid


def sample(n_steps=4, n_paths=3, dim=2, seed=0):
    return WienerProcess(dim).sample_path(
        uniform_grid(1.0, n_steps), n_paths=n_paths, seed=seed
    )


# --------------------------------------------------------------- WienerProcess


def test_wiener_process_is_a_noise_process():
    W = WienerProcess(3)
    assert isinstance(W, NoiseProcess)
    assert W.dim == 3
    assert repr(W) == "WienerProcess(dim=3)"


@pytest.mark.parametrize(
    ("dim", "error"),
    [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError)],
)
def test_wiener_process_rejects_bad_dimension(dim, error):
    with pytest.raises(error, match="dim"):
        WienerProcess(dim)


@pytest.mark.parametrize(
    ("n_paths", "error"), [(0, ValueError), (2.5, TypeError), (True, TypeError)]
)
def test_sample_path_rejects_bad_path_count(n_paths, error):
    with pytest.raises(error, match="n_paths"):
        WienerProcess().sample_path([0.0, 1.0], n_paths=n_paths)


def test_sample_path_rejects_integer_dtype():
    with pytest.raises(TypeError, match="real floating-point"):
        WienerProcess().sample_path([0.0, 1.0], seed=1, dtype=np.int64)


def test_sample_path_needs_two_grid_points():
    with pytest.raises(ValueError, match="at least 2"):
        WienerProcess().sample_path([0.0])


def test_sample_path_shapes_and_start_value():
    path = sample(n_steps=5, n_paths=4, dim=3)
    assert path.values.shape == (6, 4, 3)
    assert path.increments.shape == (5, 4, 3)
    assert (path.n_steps, path.n_paths, path.dim) == (5, 4, 3)
    assert np.all(path.values[0] == 0.0)
    assert (path.t0, path.t_end) == (0.0, 1.0)
    np.testing.assert_allclose(path.step_sizes, 0.2)


def test_sample_path_is_reproducible():
    np.testing.assert_array_equal(sample(seed=5).values, sample(seed=5).values)
    assert not np.array_equal(sample(seed=5).values, sample(seed=6).values)


def test_sample_path_accepts_backend_name_and_module():
    from numsde._backend import namespace_for

    by_name = WienerProcess().sample_path([0.0, 1.0], seed=1, backend="numpy")
    by_module = WienerProcess().sample_path(
        [0.0, 1.0], seed=1, backend=namespace_for("numpy")
    )
    np.testing.assert_array_equal(by_name.values, by_module.values)


def test_sample_path_dtype():
    path = WienerProcess().sample_path([0.0, 1.0], seed=1, dtype=np.float32)
    assert path.values.dtype == np.float32


def test_increments_match_scaled_normals():
    # W(t_n) = cumsum(sqrt(h) Z) with the generator's first draws.
    path = WienerProcess(2).sample_path([0.0, 0.25, 1.0], n_paths=3, seed=9)
    z = RandomNumberGenerator(9).standard_normal((2, 3, 2))
    expected = np.sqrt(np.array([0.25, 0.75]))[:, None, None] * z
    np.testing.assert_allclose(path.increments, expected)


# ------------------------------------------------------------- BrownianPath


def test_constructor_validates_shape():
    with pytest.raises(ValueError, match="values must have shape"):
        BrownianPath([0.0, 1.0], np.zeros((3, 1, 1)))
    with pytest.raises(ValueError, match="values must have shape"):
        BrownianPath([0.0, 1.0], np.zeros((2, 1)))


def test_constructor_wraps_existing_values():
    values = np.array([[[0.0]], [[0.5]], [[-0.25]]])
    path = BrownianPath([0.0, 1.0, 2.0], values, seed=3)
    np.testing.assert_array_equal(path.increments[:, 0, 0], [0.5, -0.75])
    assert path.seed_sequence.entropy == 3
    assert repr(path) == (
        "BrownianPath(t0=0.0, t_end=2.0, n_steps=2, n_paths=1, dim=1)"
    )


def test_coarsen_sums_increments():
    fine = sample(n_steps=12)
    coarse = fine.coarsen(3)
    np.testing.assert_array_equal(coarse.times, fine.times[::3])
    summed = fine.increments.reshape(4, 3, *fine.increments.shape[1:]).sum(axis=1)
    np.testing.assert_allclose(coarse.increments, summed)
    assert coarse.seed_sequence.entropy == fine.seed_sequence.entropy


def test_coarsen_by_one_is_identity():
    path = sample()
    np.testing.assert_array_equal(path.coarsen(1).values, path.values)


@pytest.mark.parametrize(
    ("factor", "error"),
    [
        (0, ValueError),
        (-2, ValueError),
        (1.5, TypeError),
        (2.0, TypeError),
        (True, TypeError),
    ],
)
@pytest.mark.parametrize("method", ["coarsen", "refine"])
def test_grid_operations_reject_bad_factor(method, factor, error):
    with pytest.raises(error, match="factor"):
        getattr(sample(), method)(factor)


def test_coarsen_rejects_non_divisor():
    with pytest.raises(ValueError, match="does not divide"):
        sample(n_steps=4).coarsen(3)


def test_refine_then_coarsen_is_exact():
    path = sample(n_steps=3)
    fine = path.refine(4)
    assert fine.n_steps == 12
    np.testing.assert_allclose(fine.times[::4], path.times)
    np.testing.assert_array_equal(fine.coarsen(4).values, path.values)


def test_refine_by_one_returns_same_path():
    path = sample()
    assert path.refine(1) is path


def test_refine_with_explicit_seed_is_reproducible():
    path = sample()
    a = path.refine(2, seed=11).values
    b = path.refine(2, seed=11).values
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, path.refine(2).values)


def test_insert_keeps_old_values_and_sorts():
    path = sample(n_steps=2)
    merged = path.insert([0.9, 0.1, 0.11, 0.5, 3.0, 2.0])
    np.testing.assert_allclose(merged.times, [0.0, 0.1, 0.11, 0.5, 0.9, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(merged.values[[0, 3, 5]], path.values)


def test_insert_existing_times_returns_same_path():
    path = sample()
    assert path.insert([0.0, 0.5, 1.0]) is path
    assert path.insert([]) is path


def test_insert_rejects_times_before_start():
    with pytest.raises(ValueError, match="before t0"):
        sample().insert([-0.1])


def test_insert_rejects_non_finite_times():
    with pytest.raises(ValueError, match="finite"):
        sample().insert([np.inf])


def test_insert_into_single_point_path():
    start = BrownianPath([0.0], np.zeros((1, 2, 1)), seed=4)
    path = start.insert([1.0, 0.5])
    assert path.n_steps == 2
    assert np.all(path.values[0] == 0.0)
