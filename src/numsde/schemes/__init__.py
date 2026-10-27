"""Numerical schemes, one class per update rule."""

from numsde.schemes.base import OneStepScheme, Scheme, matvec
from numsde.schemes.explicit import EulerMaruyama

__all__ = ["EulerMaruyama", "OneStepScheme", "Scheme", "matvec"]
