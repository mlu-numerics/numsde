"""Ownership and dtype checks for arrays across backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import array_api_compat
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

__all__ = [
    "asarray_in",
    "asarray_like",
    "device_of",
    "is_real_floating",
    "owned_read_only",
]


def device_of(x: Any) -> Any:
    """Return the device of array `x` (``"cpu"`` for NumPy)."""
    return array_api_compat.device(x)


def asarray_in(value: Any, xp: ModuleType, *, dtype: Any, device: Any) -> Any:
    """Convert `value` to an array of namespace `xp` with the given dtype and device.

    Read-only NumPy arrays (such as frozen problem parameters) are copied
    before they are handed to another library, which may not support
    non-writeable memory.

    Parameters
    ----------
    value
        Array-like data.
    xp
        Target Array API namespace.
    dtype, device
        Target dtype and device.

    Returns
    -------
    array
        `value` in namespace `xp`.

    Examples
    --------
    >>> from numsde._backend import namespace_for
    >>> asarray_in([1, 2], namespace_for("numpy"), dtype=None, device=None)
    array([1, 2])
    """
    if (
        isinstance(value, np.ndarray)
        and not value.flags.writeable
        and not array_api_compat.is_numpy_namespace(xp)
    ):
        value = value.copy()
    return xp.asarray(value, dtype=dtype, device=device)


def asarray_like(value: Any, like: Any) -> Any:
    """Convert `value` to an array with the namespace, dtype and device of `like`.

    Problem classes use this to turn parameters such as matrices, stored as
    NumPy arrays or Python numbers, into arrays that can be combined with the
    state `x`, whatever backend and device the simulation runs on.

    Parameters
    ----------
    value
        Array-like data: a Python number, a nested sequence or an array.
    like
        Array whose namespace, dtype and device the result takes.

    Returns
    -------
    array
        `value` in the namespace of `like`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.zeros(2, dtype=np.float32)
    >>> asarray_like([[1, 2], [3, 4]], x).dtype
    dtype('float32')
    """
    xp = array_api_compat.array_namespace(like)
    return asarray_in(value, xp, dtype=like.dtype, device=device_of(like))


def owned_read_only(x: Any, xp: ModuleType, *, copy: bool) -> Any:
    """Return an array that no other object can modify.

    Parameters
    ----------
    x
        Array in namespace `xp`.
    xp
        Array API namespace of `x`.
    copy
        Copy `x` first. Pass ``False`` only for arrays that were just created
        and are referenced nowhere else.

    Returns
    -------
    array
        For NumPy, an array backed by an immutable :class:`bytes` buffer. Its
        ``writeable`` flag is cleared and can be set again neither on the
        array nor on any view of it, because no writeable memory exists.
        NumPy arrays are always copied into that buffer, whatever `copy` is.
        Other backends have no read-only arrays, so the result is only
        protected against aliasing.

    Examples
    --------
    >>> import numpy as np
    >>> from numsde._backend import namespace_for
    >>> frozen = owned_read_only(np.ones(2), namespace_for("numpy"), copy=True)
    >>> frozen.flags.writeable = True
    Traceback (most recent call last):
    ...
    ValueError: cannot set WRITEABLE flag to True of this array
    """
    if isinstance(x, np.ndarray):
        # A writeable flag cleared on an array that owns its memory can be set
        # again. An array whose memory is an immutable bytes object cannot.
        buffer = np.ascontiguousarray(x).tobytes()
        return np.frombuffer(buffer, dtype=x.dtype).reshape(x.shape)
    if copy:
        x = xp.asarray(x, copy=True)
    return x


def is_real_floating(x: Any, xp: ModuleType) -> bool:
    """Return whether `x` has a real floating-point dtype."""
    return bool(xp.isdtype(x.dtype, "real floating"))
