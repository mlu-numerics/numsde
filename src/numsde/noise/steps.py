r"""Step quantities and noise sources.

A scheme never sees a whole noise path. On each step :math:`[t_n, t_{n+1}]`
it receives a :class:`StepNoise` with exactly the random quantities it needs,
for Euler–Maruyama the step size :math:`h_n` and the increment
:math:`\Delta W_n`. A :class:`NoiseSource` produces these objects step by
step; :class:`~numsde.BrownianPath` is the materialized implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from types import ModuleType

    import numpy as np
    import numpy.typing as npt

__all__ = ["STEP_QUANTITIES", "NoiseSource", "StepNoise"]

STEP_QUANTITIES: Final[tuple[str, ...]] = ("dW", "I", "Z")
r"""Names of the random quantities a scheme may request: the increment
:math:`\Delta W_n`, the double Itô integrals :math:`I_{ij}` and
:math:`Z^j = I_{(j,0)}`."""


@dataclass(frozen=True)
class StepNoise:
    r"""All random quantities of one step :math:`[t_n, t_{n+1}]`.

    Parameters
    ----------
    t
        Left end point :math:`t_n`.
    h
        Step size :math:`h_n = t_{n+1} - t_n`.
    dW
        Increments :math:`\Delta W_n = W(t_{n+1}) - W(t_n)`, shape
        ``(n_paths, m)``.
    I
        Double Itô integrals :math:`I_{ij}(t_n, t_{n+1})`, shape
        ``(n_paths, m, m)``, if requested. No source provides them yet
        (milestone M2).
    Z
        Integrals :math:`Z^j(t_n, t_{n+1})`, shape ``(n_paths, m)``, if
        requested. No source provides them yet (milestone M2).

    Examples
    --------
    >>> import numpy as np
    >>> noise = StepNoise(t=0.0, h=0.5, dW=np.array([[0.1]]))
    >>> noise.dW.shape
    (1, 1)
    """

    t: float
    h: float
    dW: Any  # noqa: N815 - the standard symbol for Brownian increments
    I: Any = None  # noqa: E741 - the standard symbol for iterated integrals
    Z: Any = None


class NoiseSource(ABC):
    """A realization of the driving noise, delivered step by step.

    A noise source is one fixed realization on a fixed time grid: iterating
    over :meth:`steps` twice yields the same quantities. Solvers depend only
    on this interface, so a materialized :class:`~numsde.BrownianPath` and a
    streaming source are interchangeable.
    """

    provides: ClassVar[frozenset[str]] = frozenset({"dW"})
    """Step quantities this source can compute."""

    @property
    @abstractmethod
    def times(self) -> npt.NDArray[np.float64]:
        """Grid points (read-only NumPy array)."""

    @property
    @abstractmethod
    def n_paths(self) -> int:
        """Number of independent paths."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension :math:`m` of the noise."""

    @property
    @abstractmethod
    def namespace(self) -> ModuleType:
        """Array API namespace of the produced arrays."""

    @property
    @abstractmethod
    def dtype(self) -> Any:
        """Dtype of the produced arrays."""

    @property
    @abstractmethod
    def device(self) -> Any:
        """Device of the produced arrays."""

    @property
    def n_steps(self) -> int:
        """Number of steps of the grid."""
        return int(self.times.size) - 1

    def steps(self, requires: Iterable[str] = ("dW",)) -> Iterator[StepNoise]:
        """Yield the step quantities of every step, in time order.

        Parameters
        ----------
        requires
            Names of the quantities needed, a subset of :attr:`provides`.

        Yields
        ------
        StepNoise
            One object per step; quantities not requested are ``None``.

        Raises
        ------
        ValueError
            If a requested quantity is unknown or not provided.
        """
        requested = self.check_requires(requires)
        yield from self._steps(requested)

    def check_requires(self, requires: Iterable[str]) -> frozenset[str]:
        """Return `requires` as a set after checking that it can be provided.

        Parameters
        ----------
        requires
            Names of step quantities.

        Returns
        -------
        frozenset of str
            The requested names.

        Raises
        ------
        ValueError
            If a name is unknown or not in :attr:`provides`.
        """
        requested = frozenset(requires)
        unknown = requested - set(STEP_QUANTITIES)
        if unknown:
            msg = (
                f"Unknown step quantities {sorted(unknown)}; "
                f"known quantities are {list(STEP_QUANTITIES)}."
            )
            raise ValueError(msg)
        missing = requested - self.provides
        if missing:
            msg = (
                f"{type(self).__name__} cannot provide the step quantities "
                f"{sorted(missing)}; it provides {sorted(self.provides)}."
            )
            raise ValueError(msg)
        return requested

    @abstractmethod
    def _steps(self, requires: frozenset[str]) -> Iterator[StepNoise]:
        """Yield the step quantities; `requires` has been checked."""
