"""Resolution of the array namespace used by a computation."""

from __future__ import annotations

import importlib
import numbers
from typing import TYPE_CHECKING, Final

import array_api_compat

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ["SUPPORTED_BACKENDS", "array_namespace", "namespace_for"]

SUPPORTED_BACKENDS: Final[tuple[str, ...]] = ("numpy", "torch")
"""Names accepted by :func:`namespace_for`."""

_INSTALL_HINT: Final[dict[str, str]] = {
    "numpy": "pip install numpy",
    "torch": "pip install 'numsde[torch]'",
}


def array_namespace(*arrays: object) -> ModuleType:
    """Return the Array API namespace shared by the given arrays.

    Python scalars and ``None`` are ignored, so mixed calls such as
    ``array_namespace(x, 0.5)`` work. Without any array argument the NumPy
    namespace is returned, which makes NumPy the default backend.

    Parameters
    ----------
    *arrays
        Arrays (NumPy, PyTorch, ...), Python scalars or ``None``.

    Returns
    -------
    types.ModuleType
        An Array API compatible namespace, e.g. ``array_api_compat.numpy``.

    Raises
    ------
    TypeError
        If the arguments belong to different array libraries, or if an
        argument is neither an array, a Python scalar nor ``None``.

    Examples
    --------
    >>> import numpy as np
    >>> xp = array_namespace(np.ones(3), 2.0)
    >>> xp.__name__
    'array_api_compat.numpy'
    >>> array_namespace().__name__
    'array_api_compat.numpy'
    """
    candidates = [
        a for a in arrays if a is not None and not isinstance(a, numbers.Number)
    ]
    if not candidates:
        return namespace_for("numpy")
    namespace: ModuleType = array_api_compat.array_namespace(*candidates)
    return namespace


def namespace_for(backend: str) -> ModuleType:
    """Return the Array API namespace for a backend given by name.

    Parameters
    ----------
    backend
        One of :data:`SUPPORTED_BACKENDS`.

    Returns
    -------
    types.ModuleType
        The corresponding ``array_api_compat`` namespace.

    Raises
    ------
    ValueError
        If `backend` is not supported.
    ImportError
        If the library behind `backend` is not installed.

    Examples
    --------
    >>> namespace_for("numpy").__name__
    'array_api_compat.numpy'
    """
    if backend not in SUPPORTED_BACKENDS:
        supported = ", ".join(repr(b) for b in SUPPORTED_BACKENDS)
        msg = f"Unknown backend {backend!r}; supported backends are {supported}."
        raise ValueError(msg)
    try:
        namespace = importlib.import_module(f"array_api_compat.{backend}")
    except ImportError as exc:
        msg = (
            f"The {backend!r} backend is not available. "
            f"Install it with `{_INSTALL_HINT[backend]}`."
        )
        raise ImportError(msg) from exc
    return namespace
