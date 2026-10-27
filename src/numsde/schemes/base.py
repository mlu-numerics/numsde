"""Base classes of numerical schemes.

A scheme is one update rule. Its instances hold parameters only, so they can
be reused across runs and threads; state that a run needs (for example the
history of a multistep scheme) lives in the object returned by
:meth:`Scheme.init_state`, which the solver owns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from numsde._backend import array_namespace
from numsde.problems.structure import Calculus

if TYPE_CHECKING:
    from numsde.noise.steps import StepNoise
    from numsde.problems.base import SDEProblem
    from numsde.problems.structure import NoiseStructure

__all__ = ["OneStepScheme", "Scheme", "matvec"]


def matvec(A: Any, v: Any) -> Any:
    r"""Batched matrix-vector product :math:`A v`.

    Parameters
    ----------
    A
        Matrices of shape ``(..., d, m)``.
    v
        Vectors of shape ``(..., m)``; leading axes broadcast against `A`.

    Returns
    -------
    array
        Shape ``(..., d)``.

    Examples
    --------
    >>> import numpy as np
    >>> matvec(np.eye(2)[None] * 2.0, np.array([[1.0, 3.0]]))
    array([[2., 6.]])
    """
    xp = array_namespace(A, v)
    return xp.matmul(A, v[..., None])[..., 0]


class Scheme(ABC):
    """Base class of all schemes.

    Subclasses declare what they need and which orders they achieve through
    class attributes; the solver checks the requirements before the time
    loop starts.
    """

    name: ClassVar[str]
    """Human-readable name of the scheme."""

    strong_order: ClassVar[float]
    """Strong order under the standard assumptions of its theory page."""

    weak_order: ClassVar[float | None] = None
    """Weak order under the standard assumptions, if known."""

    requires: ClassVar[frozenset[str]] = frozenset({"dW"})
    """Step quantities the scheme reads from :class:`~numsde.StepNoise`."""

    calculi: ClassVar[frozenset[Calculus]] = frozenset({Calculus.ITO})
    """Calculi the scheme is consistent with."""

    requires_structure: ClassVar[frozenset[str]] = frozenset()
    """Noise-structure flags the problem must have."""

    def expected_strong_order(self, structure: NoiseStructure) -> float:
        """Return the strong order for a problem with the given noise structure.

        Parameters
        ----------
        structure
            Noise structure of the problem.

        Returns
        -------
        float
            :attr:`strong_order`; schemes whose order improves for special
            noise override this.
        """
        return self.strong_order

    def check_problem(self, problem: SDEProblem) -> None:
        """Raise if the scheme cannot be applied to `problem`.

        Parameters
        ----------
        problem
            The problem to be solved.

        Raises
        ------
        ValueError
            If the calculus or the noise structure of `problem` is not
            supported.
        """
        if problem.calculus not in self.calculi:
            supported = ", ".join(sorted(c.value for c in self.calculi))
            msg = (
                f"{self.name} supports {supported} calculus, but the problem is "
                f"interpreted in {problem.calculus.value} calculus."
            )
            raise ValueError(msg)
        missing = self.requires_structure - problem.structure.flags
        if missing:
            msg = (
                f"{self.name} requires noise structure {sorted(missing)}, which "
                f"the problem does not declare ({problem.structure!r})."
            )
            raise ValueError(msg)

    def init_state(self, problem: SDEProblem, x0: Any) -> Any:
        """Return the run state for a new simulation (``None`` by default).

        Parameters
        ----------
        problem
            The problem to be solved.
        x0
            Initial values of the batch, shape ``(n_paths, dim)``.

        Returns
        -------
        object
            State passed to every call of :meth:`step`.
        """
        return None

    @abstractmethod
    def step(
        self, problem: SDEProblem, t: float, x: Any, noise: StepNoise, state: Any
    ) -> Any:
        """Advance the batch `x` from ``t`` to ``t + noise.h``.

        Parameters
        ----------
        problem
            The problem.
        t
            Current time :math:`t_n`.
        x
            Current values :math:`X_h^n`, shape ``(n_paths, dim)``.
        noise
            Random quantities of the step.
        state
            The object returned by :meth:`init_state`; multistep schemes
            update it in place.

        Returns
        -------
        array
            :math:`X_h^{n+1}`, shape ``(n_paths, dim)``, a new array.
        """

    def __repr__(self) -> str:
        """Return the class name."""
        return f"{type(self).__name__}()"


class OneStepScheme(Scheme):
    """Base class of one-step schemes, which need no run state."""
