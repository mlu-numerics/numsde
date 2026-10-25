"""Reproducible random number generation independent of the array backend.

Samples are drawn with NumPy's PCG64 generator and converted to the target
array namespace afterwards. The same seed therefore yields the same numbers
on every backend, which keeps results comparable across NumPy and PyTorch.
Backend-native sampling (e.g. directly on a GPU) may be added later as an
optimisation behind the same interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ["RandomNumberGenerator", "SeedLike", "default_rng"]

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
