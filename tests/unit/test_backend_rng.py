import numpy as np
import pytest

from numsde import RandomNumberGenerator, default_rng


def test_same_seed_same_samples():
    a = RandomNumberGenerator(123).standard_normal((4, 5))
    b = RandomNumberGenerator(123).standard_normal((4, 5))
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ():
    a = RandomNumberGenerator(1).standard_normal(100)
    b = RandomNumberGenerator(2).standard_normal(100)
    assert not np.array_equal(a, b)


def test_default_output_is_numpy_float64():
    z = RandomNumberGenerator(0).standard_normal((3,))
    assert isinstance(z, np.ndarray)
    assert z.dtype == np.float64


def test_dtype_conversion():
    z = RandomNumberGenerator(0).standard_normal(3, dtype=np.float32)
    assert z.dtype == np.float32


def test_namespace_conversion_keeps_values():
    from numsde._backend import namespace_for

    xp = namespace_for("numpy")
    ref = RandomNumberGenerator(5).standard_normal(6)
    z = RandomNumberGenerator(5).standard_normal(6, xp=xp)
    np.testing.assert_array_equal(z, ref)
    assert z.dtype == np.float64


def test_device_is_passed_through():
    z = RandomNumberGenerator(0).standard_normal(2, device="cpu")
    assert z.shape == (2,)


def test_seed_sequence_is_reused():
    ss = np.random.SeedSequence(99)
    rng = RandomNumberGenerator(ss)
    assert rng.seed_sequence is ss


def test_none_seed_uses_fresh_entropy():
    rng = RandomNumberGenerator()
    assert rng.seed_sequence.entropy is not None


def test_spawn_is_reproducible_and_independent():
    first = [r.standard_normal(10) for r in RandomNumberGenerator(3).spawn(3)]
    second = [r.standard_normal(10) for r in RandomNumberGenerator(3).spawn(3)]
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[0], first[1])


def test_spawn_zero():
    assert RandomNumberGenerator(0).spawn(0) == []


def test_default_rng_passthrough_and_seed():
    rng = default_rng(5)
    assert isinstance(rng, RandomNumberGenerator)
    assert default_rng(rng) is rng


def test_repr_shows_entropy():
    assert repr(RandomNumberGenerator(17)) == "RandomNumberGenerator(entropy=17)"


def test_torch_samples_match_numpy():
    torch = pytest.importorskip("torch")
    from numsde._backend import namespace_for

    xp = namespace_for("torch")
    z = RandomNumberGenerator(11).standard_normal((2, 2), xp=xp)
    assert isinstance(z, torch.Tensor)
    assert z.dtype == torch.float64
    ref = RandomNumberGenerator(11).standard_normal((2, 2))
    np.testing.assert_array_equal(z.numpy(), ref)
