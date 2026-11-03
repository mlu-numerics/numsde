r"""Structural properties of the diffusion coefficient.

The structure of the diffusion :math:`g` decides which schemes apply and which
fast code paths they may take. The properties overlap, so they are combinable
flags rather than an exclusive enum:

``scalar``
    :math:`m = 1`.
``additive``
    :math:`g(t, x)` does not depend on :math:`x`.
``diagonal``
    :math:`d = m`, :math:`g` is diagonal and :math:`g_{ii}` depends on
    :math:`x_i` only.
``commutative``
    :math:`L^i g^j = L^j g^i` for all :math:`i, j`, where
    :math:`L^j = \sum_k g_{kj}\,\partial_{x_k}` and :math:`g^j` is the
    :math:`j`-th column of :math:`g`.

Scalar, additive and diagonal noise are commutative, and the implied flags are
added automatically. Every flag is either *declared* (asserted by the author
of a problem) or *verified* (established by the library, for example from the
dimensions or by construction). Schemes may rely on declared flags; the
solution metadata reports which flags were verified.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["FLAGS", "Calculus", "NoiseStructure"]

FLAGS: Final[tuple[str, ...]] = ("scalar", "additive", "diagonal", "commutative")
"""Names of the noise-structure flags."""

_IMPLIES: Final[dict[str, tuple[str, ...]]] = {
    "scalar": ("commutative",),
    "additive": ("commutative",),
    "diagonal": ("commutative",),
    "commutative": (),
}


class Calculus(enum.Enum):
    """Stochastic calculus in which an SDE is interpreted.

    Only Itô calculus is implemented; ``STRATONOVICH`` will be added without
    changing existing schemes, which declare the calculus they support.

    Examples
    --------
    >>> Calculus.ITO.value
    'ito'
    """

    ITO = "ito"


def _closure(flags: Iterable[str]) -> frozenset[str]:
    """Add every flag implied by `flags`."""
    result = set(flags)
    for flag in list(result):
        result.update(_IMPLIES[flag])
    return frozenset(result)


def _check_flag_names(flags: Iterable[object], name: str) -> frozenset[str]:
    names = frozenset(flags)
    for flag in names:
        if not isinstance(flag, str):
            msg = f"{name} must contain flag names, got {flag!r}."
            raise TypeError(msg)
    checked = frozenset(str(flag) for flag in names)
    unknown = checked - set(FLAGS)
    if unknown:
        msg = (
            f"Unknown noise-structure flag(s) {sorted(unknown)} in {name}; "
            f"known flags are {list(FLAGS)}."
        )
        raise ValueError(msg)
    return checked


@dataclass(frozen=True)
class NoiseStructure:
    """Combinable flags describing the structure of the diffusion.

    Parameters
    ----------
    scalar, additive, diagonal, commutative
        Flags declared for the problem. Implied flags (``commutative`` for
        each of the others) are added automatically.
    verified
        Names of flags the library has established rather than merely
        declared. Problem authors normally leave this empty; it is filled by
        the library, e.g. ``scalar`` when the noise dimension is 1, or by a
        problem whose construction guarantees a property. A flag implied by
        a verified flag is verified as well.

    Raises
    ------
    TypeError
        If a flag is not a bool.
    ValueError
        If `verified` names an unknown flag or a flag that is not set.

    Examples
    --------
    >>> s = NoiseStructure(additive=True)
    >>> s.commutative
    True
    >>> sorted(s.flags), sorted(s.verified)
    (['additive', 'commutative'], [])
    >>> sorted(s.with_verified("additive").verified)
    ['additive', 'commutative']
    """

    scalar: bool = False
    additive: bool = False
    diagonal: bool = False
    commutative: bool = False
    verified: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate the flags and add implied ones."""
        for flag in FLAGS:
            value = getattr(self, flag)
            if not isinstance(value, bool):
                msg = f"{flag} must be a bool, got {value!r}."
                raise TypeError(msg)
        declared = _closure(f for f in FLAGS if getattr(self, f))
        for flag in declared:
            object.__setattr__(self, flag, True)
        verified = _closure(_check_flag_names(self.verified, "verified"))
        missing = verified - declared
        if missing:
            msg = f"Cannot verify flag(s) {sorted(missing)} that are not set."
            raise ValueError(msg)
        object.__setattr__(self, "verified", verified)

    @property
    def flags(self) -> frozenset[str]:
        """Names of all set flags, declared or verified."""
        return frozenset(f for f in FLAGS if getattr(self, f))

    @property
    def declared_only(self) -> frozenset[str]:
        """Names of the set flags that have not been verified."""
        return self.flags - self.verified

    def with_verified(self, *flags: str) -> NoiseStructure:
        """Return a copy in which `flags` are set and marked as verified.

        Parameters
        ----------
        *flags
            Flag names.

        Returns
        -------
        NoiseStructure
            The extended structure.

        Examples
        --------
        >>> NoiseStructure().with_verified("scalar").scalar
        True
        """
        names = _check_flag_names(flags, "flags")
        return NoiseStructure(
            **{f: f in names or getattr(self, f) for f in FLAGS},
            verified=self.verified | names,
        )

    def __repr__(self) -> str:
        """Return the set flags and the verified ones."""
        set_flags = ", ".join(f"{f}=True" for f in FLAGS if getattr(self, f))
        if self.verified:
            verified = ", ".join(repr(f) for f in FLAGS if f in self.verified)
            set_flags += f"{', ' if set_flags else ''}verified={{{verified}}}"
        return f"NoiseStructure({set_flags})"
