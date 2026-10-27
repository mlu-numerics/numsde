"""Ownership and dtype checks for arrays across backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ["is_real_floating", "owned_read_only"]


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
