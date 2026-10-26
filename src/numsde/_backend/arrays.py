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
        For NumPy, an array whose ``writeable`` flag is cleared; views of it
        cannot be made writeable again. Other backends have no read-only
        arrays, so the result is only protected against aliasing.
    """
    if copy:
        x = xp.asarray(x, copy=True)
    if isinstance(x, np.ndarray):
        x.flags.writeable = False
    return x


def is_real_floating(x: Any, xp: ModuleType) -> bool:
    """Return whether `x` has a real floating-point dtype."""
    return bool(xp.isdtype(x.dtype, "real floating"))
