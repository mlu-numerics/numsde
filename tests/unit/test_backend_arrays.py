import numpy as np
import pytest

from numsde._backend import is_real_floating, namespace_for, owned_read_only


def test_numpy_arrays_become_read_only_copies():
    xp = namespace_for("numpy")
    source = np.ones(3)
    owned = owned_read_only(source, xp, copy=True)
    assert not owned.flags.writeable
    assert not np.shares_memory(owned, source)
    assert source.flags.writeable


@pytest.mark.parametrize("copy", [True, False])
def test_numpy_results_cannot_be_made_writeable_again(copy):
    # Re-audit B1: clearing the flag on an array that owns its memory is not
    # enough, because the owner may set it again.
    owned = owned_read_only(
        np.arange(6.0).reshape(2, 3), namespace_for("numpy"), copy=copy
    )
    for array in (owned, owned[1:], owned.T, owned.base):
        with pytest.raises(ValueError, match="cannot set WRITEABLE"):
            array.flags.writeable = True


def test_numpy_results_keep_dtype_shape_and_values():
    source = np.asfortranarray(np.arange(6, dtype=np.float32).reshape(2, 3))
    owned = owned_read_only(source, namespace_for("numpy"), copy=False)
    assert owned.dtype == np.float32
    assert owned.shape == (2, 3)
    np.testing.assert_array_equal(owned, source)
    assert source.flags.writeable


def test_empty_numpy_arrays_are_supported():
    owned = owned_read_only(np.empty((0, 3)), namespace_for("numpy"), copy=True)
    assert owned.shape == (0, 3)
    assert not owned.flags.writeable


def test_non_numpy_arrays_are_passed_through():
    # Backends without read-only arrays (e.g. PyTorch) are only copied.
    class Opaque:
        pass

    marker = Opaque()
    assert owned_read_only(marker, namespace_for("numpy"), copy=False) is marker


def test_non_numpy_arrays_are_copied_on_request():
    class Namespace:
        @staticmethod
        def asarray(x, *, copy):
            assert copy
            return ("copy of", x)

    assert owned_read_only("tensor", Namespace, copy=True) == ("copy of", "tensor")


def test_is_real_floating():
    xp = namespace_for("numpy")
    assert is_real_floating(np.ones(1, dtype=np.float32), xp)
    assert not is_real_floating(np.ones(1, dtype=np.int32), xp)
    assert not is_real_floating(np.ones(1, dtype=np.complex64), xp)


def test_asarray_in_copies_read_only_arrays_for_other_libraries():
    import types

    from numsde._backend import asarray_in

    fake = types.ModuleType("fake_backend")
    fake.asarray = lambda value, dtype=None, device=None: value
    frozen = np.ones(3)
    frozen.flags.writeable = False
    converted = asarray_in(frozen, fake, dtype=None, device=None)
    assert converted is not frozen
    assert converted.flags.writeable
    # NumPy itself keeps read-only arrays as they are.
    same = asarray_in(frozen, namespace_for("numpy"), dtype=None, device=None)
    assert not same.flags.writeable


def test_asarray_like_follows_dtype_of_the_reference():
    from numsde._backend import asarray_like

    x = np.zeros(2, dtype=np.float32)
    assert asarray_like([1.0, 2.0], x).dtype == np.float32
