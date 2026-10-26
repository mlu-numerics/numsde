"""Reproducible random number generation independent of the array backend.

Samples are drawn with NumPy's PCG64 generator and converted to the target
array namespace afterwards. The same seed therefore yields the same numbers
on every backend, which keeps results comparable across NumPy and PyTorch.
Backend-native sampling (e.g. directly on a GPU) may be added later as an
optimization behind the same interface.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

__all__ = [
    "RandomNumberGenerator",
    "SeedLike",
    "default_rng",
    "derive_seed_sequence",
    "frozen_seed_sequence",
]

SeedLike: TypeAlias = int | Sequence[int] | np.random.SeedSequence | None
"""Values accepted as a seed: an integer, a sequence of integers, a
:class:`numpy.random.SeedSequence`, or ``None`` for fresh OS entropy."""


class RandomNumberGenerator:
    """Seeded source of random numbers for simulations.

    Parameters
    ----------
    seed
        Seed for the underlying :class:`numpy.random.SeedSequence`. ``None``
        draws fresh entropy from the operating system.

    Examples
    --------
    >>> rng = RandomNumberGenerator(seed=42)
    >>> z = rng.standard_normal((2, 3))
    >>> z.shape
    (2, 3)
    >>> bool((RandomNumberGenerator(42).standard_normal((2, 3)) == z).all())
    True
    """

    def __init__(self, seed: SeedLike = None) -> None:
        if isinstance(seed, np.random.SeedSequence):
            self._seed_sequence = seed
        else:
            self._seed_sequence = np.random.SeedSequence(seed)
        self._generator = np.random.Generator(np.random.PCG64(self._seed_sequence))

    @property
    def seed_sequence(self) -> np.random.SeedSequence:
        """The seed sequence this generator was created from."""
        return self._seed_sequence

    def standard_normal(
        self,
        shape: int | tuple[int, ...],
        *,
        xp: ModuleType | None = None,
        dtype: Any = None,
        device: Any = None,
    ) -> Any:
        """Draw independent standard normal samples.

        Parameters
        ----------
        shape
            Shape of the returned array.
        xp
            Target Array API namespace. Defaults to NumPy.
        dtype
            Floating-point dtype of the target namespace. Defaults to
            double precision.
        device
            Target device, passed through to ``xp.asarray``.

        Returns
        -------
        array
            Array of the requested shape in namespace `xp`.

        Examples
        --------
        >>> RandomNumberGenerator(0).standard_normal(4).dtype
        dtype('float64')
        """
        samples = self._generator.standard_normal(shape)
        if xp is None and dtype is None and device is None:
            return samples
        if xp is None:
            xp = np
        if dtype is None:
            dtype = xp.float64
        return xp.asarray(samples, dtype=dtype, device=device)

    def spawn(self, n: int) -> list[RandomNumberGenerator]:
        """Create `n` statistically independent child generators.

        Use this to give each parallel worker or each simulation level its
        own stream; the children are reproducible from the parent seed.

        Parameters
        ----------
        n
            Number of children, at least zero.

        Returns
        -------
        list of RandomNumberGenerator
            The child generators.

        Examples
        --------
        >>> workers = RandomNumberGenerator(7).spawn(3)
        >>> len(workers)
        3
        """
        return [RandomNumberGenerator(child) for child in self._seed_sequence.spawn(n)]

    def __repr__(self) -> str:
        """Return a representation showing the seed entropy."""
        return f"RandomNumberGenerator(entropy={self._seed_sequence.entropy!r})"


def default_rng(seed: SeedLike | RandomNumberGenerator = None) -> RandomNumberGenerator:
    """Return a :class:`RandomNumberGenerator`.

    An existing generator is passed through unchanged, so functions can
    accept either a seed or a generator.

    Parameters
    ----------
    seed
        A seed (an integer, a sequence of integers, a
        :class:`numpy.random.SeedSequence` or ``None``) or an existing generator.

    Returns
    -------
    RandomNumberGenerator
        The generator.

    Examples
    --------
    >>> rng = default_rng(1)
    >>> default_rng(rng) is rng
    True
    """
    if isinstance(seed, RandomNumberGenerator):
        return seed
    return RandomNumberGenerator(seed)


def frozen_seed_sequence(seed: SeedLike) -> np.random.SeedSequence:
    """Return a new seed sequence with the same entropy and spawn key as `seed`.

    :class:`numpy.random.SeedSequence` counts the children it has spawned, so
    a shared instance carries mutable state. Objects that must behave as
    values store a private copy made by this function and never spawn from
    it.

    Parameters
    ----------
    seed
        A seed or a seed sequence; ``None`` draws fresh entropy.

    Returns
    -------
    numpy.random.SeedSequence
        An independent copy.

    Examples
    --------
    >>> frozen_seed_sequence(5).entropy
    5
    """
    if isinstance(seed, np.random.SeedSequence):
        return np.random.SeedSequence(seed.entropy, spawn_key=seed.spawn_key)
    return np.random.SeedSequence(seed)


def derive_seed_sequence(
    parent: np.random.SeedSequence, data: bytes
) -> np.random.SeedSequence:
    """Derive a child seed sequence deterministically from `parent` and `data`.

    The child extends the parent's spawn key by four 32-bit words of the
    SHA-256 digest of `data`. The same parent and data always give the same
    child; different data give statistically independent streams. The parent
    is not modified.

    Parameters
    ----------
    parent
        Seed sequence to derive from.
    data
        Bytes that identify the request, e.g. the times to be sampled.

    Returns
    -------
    numpy.random.SeedSequence
        The derived seed sequence.

    Examples
    --------
    >>> root = np.random.SeedSequence(1)
    >>> a = derive_seed_sequence(root, b"x").generate_state(1)
    >>> b = derive_seed_sequence(root, b"x").generate_state(1)
    >>> bool((a == b).all())
    True
    """
    digest = hashlib.sha256(data).digest()
    words = tuple(int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4))
    return np.random.SeedSequence(parent.entropy, spawn_key=(*parent.spawn_key, *words))
