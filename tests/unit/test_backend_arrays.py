import numpy as np

from numsde._backend import is_real_floating, namespace_for, owned_read_only


def test_numpy_arrays_become_read_only_copies():
    xp = namespace_for("numpy")
    source = np.ones(3)
    owned = owned_read_only(source, xp, copy=True)
    assert not owned.flags.writeable
    assert not np.shares_memory(owned, source)
    assert source.flags.writeable


def test_without_copy_the_array_itself_is_frozen():
    xp = namespace_for("numpy")
    fresh = np.ones(3)
    assert owned_read_only(fresh, xp, copy=False) is fresh
    assert not fresh.flags.writeable


def test_non_numpy_arrays_are_passed_through():
    # Backends without read-only arrays (e.g. PyTorch) are only copied.
    class Opaque:
        pass

    marker = Opaque()
    assert owned_read_only(marker, namespace_for("numpy"), copy=False) is marker


def test_is_real_floating():
    xp = namespace_for("numpy")
    assert is_real_floating(np.ones(1, dtype=np.float32), xp)
    assert not is_real_floating(np.ones(1, dtype=np.int32), xp)
    assert not is_real_floating(np.ones(1, dtype=np.complex64), xp)
