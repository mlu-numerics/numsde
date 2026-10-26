"""Validation helpers shared by all public entry points.

Wrong types raise :class:`TypeError`, wrong values raise :class:`ValueError`.
Booleans are rejected wherever an integer is expected, although ``bool`` is a
subclass of ``int`` in Python.
"""

from __future__ import annotations

import math
import numbers

__all__ = ["finite_float", "positive_int"]


def positive_int(value: object, name: str) -> int:
    """Return `value` as an ``int`` after checking that it is a positive integer.

    Parameters
    ----------
    value
        Candidate value; Python and NumPy integers are accepted.
    name
        Parameter name used in error messages.

    Returns
    -------
    int
        The validated value.

    Raises
    ------
    TypeError
        If `value` is a bool, a float or not an integer at all.
    ValueError
        If `value` is smaller than 1.

    Examples
    --------
    >>> positive_int(3, "n_steps")
    3
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        kind = type(value).__name__
        msg = f"{name} must be an integer, got {value!r} of type {kind}."
        raise TypeError(msg)
    if value < 1:
        msg = f"{name} must be positive, got {value}."
        raise ValueError(msg)
    return int(value)


def finite_float(value: object, name: str) -> float:
    """Return `value` as a ``float`` after checking that it is a finite real number.

    Parameters
    ----------
    value
        Candidate value; Python and NumPy real numbers are accepted.
    name
        Parameter name used in error messages.

    Returns
    -------
    float
        The validated value.

    Raises
    ------
    TypeError
        If `value` is a bool or not a real number.
    ValueError
        If `value` is infinite or NaN.

    Examples
    --------
    >>> finite_float(1, "t_end")
    1.0
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        kind = type(value).__name__
        msg = f"{name} must be a real number, got {value!r} of type {kind}."
        raise TypeError(msg)
    if not math.isfinite(value):
        msg = f"{name} must be finite, got {value}."
        raise ValueError(msg)
    return float(value)
