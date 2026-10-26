"""Abstract interface of driving noise processes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from numsde._validation import positive_int

if TYPE_CHECKING:
    from types import ModuleType

    from numsde._backend import SeedLike
    from numsde.stepping.grids import TimeGridLike

__all__ = ["NoiseProcess"]


class NoiseProcess(ABC):
    """Base class of the processes that drive an SDE.

    A noise process describes the law of the driving signal, for example a
    Wiener process. Sampling it on a time grid yields a noise path. Later
    extensions such as fractional Brownian motion or Lévy processes subclass
    this interface.

    Parameters
    ----------
    dim
        Dimension :math:`m` of the process, at least 1.

    Raises
    ------
    TypeError
        If `dim` is not an integer.
    ValueError
        If `dim` is not positive.
    """

    def __init__(self, dim: int = 1) -> None:
        self._dim = positive_int(dim, "dim")

    @property
    def dim(self) -> int:
        """Dimension :math:`m` of the process."""
        return self._dim

    @abstractmethod
    def sample_path(
        self,
        times: TimeGridLike,
        n_paths: int = 1,
        *,
        seed: SeedLike = None,
        backend: str | ModuleType | None = None,
        dtype: Any = None,
        device: Any = None,
    ) -> Any:
        """Sample `n_paths` independent paths of the process on a time grid."""

    def __repr__(self) -> str:
        """Return a representation showing the dimension."""
        return f"{type(self).__name__}(dim={self._dim})"
