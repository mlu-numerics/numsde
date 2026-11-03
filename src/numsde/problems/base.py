r"""Problem classes: the initial value problem of an SDE.

A problem is the initial value problem

.. math::

    \mathrm{d}X(t) = f(t, X(t))\,\mathrm{d}t + g(t, X(t))\,\mathrm{d}W(t),
    \qquad X(t_0) = x_0,

with drift :math:`f(t, x) \in \mathbb{R}^d`, diffusion
:math:`g(t, x) \in \mathbb{R}^{d \times m}` and an :math:`m`-dimensional
Wiener process :math:`W`. Users define a problem by subclassing
:class:`ItoSDE` and implementing :meth:`~SDEProblem.drift` and
:meth:`~SDEProblem.diffusion`.
"""

from __future__ import annotations

import numbers
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import array_api_compat
import numpy as np

from numsde._backend import asarray_in, is_real_floating, owned_read_only
from numsde._validation import finite_float, instance_of, positive_int
from numsde.problems.structure import Calculus, NoiseStructure

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ["ItoSDE", "SDEProblem", "resolve_structure"]


def resolve_structure(structure: object, dim: int, noise_dim: int) -> NoiseStructure:
    """Check declared flags against the dimensions and add verifiable ones.

    Parameters
    ----------
    structure
        Declared structure.
    dim, noise_dim
        Dimensions :math:`d` and :math:`m`.

    Returns
    -------
    NoiseStructure
        `structure` with ``scalar`` set and verified if ``noise_dim == 1``.

    Raises
    ------
    TypeError
        If `structure` is not a :class:`NoiseStructure`.
    ValueError
        If ``scalar`` is declared with ``noise_dim != 1`` or ``diagonal``
        with ``dim != noise_dim``.

    Examples
    --------
    >>> resolve_structure(NoiseStructure(), dim=2, noise_dim=1)
    NoiseStructure(scalar=True, commutative=True, verified={'scalar', 'commutative'})
    """
    structure = instance_of(structure, NoiseStructure, "structure")
    if structure.scalar and noise_dim != 1:
        msg = f"Scalar noise requires noise_dim == 1, got noise_dim={noise_dim}."
        raise ValueError(msg)
    if structure.diagonal and dim != noise_dim:
        msg = (
            "Diagonal noise requires dim == noise_dim, "
            f"got dim={dim}, noise_dim={noise_dim}."
        )
        raise ValueError(msg)
    if noise_dim == 1:
        structure = structure.with_verified("scalar")
    return structure


class SDEProblem(ABC):
    r"""Base class of SDE initial value problems.

    Subclasses set the class attribute :attr:`calculus` and implement
    :meth:`drift` and :meth:`diffusion`; :class:`ItoSDE` is the subclass for
    Itô equations. A subclass may declare its noise structure with a class
    attribute ``structure = NoiseStructure(...)``.

    Parameters
    ----------
    dim
        Dimension :math:`d` of the state.
    noise_dim
        Dimension :math:`m` of the Wiener process.
    x0
        Initial value :math:`x_0`: a Python number if ``dim == 1``, a
        sequence of ``dim`` numbers, or a real floating-point array of shape
        ``(dim,)`` in any Array API namespace. If it is an array, the
        simulation runs in its namespace, dtype and device.
    t0
        Initial time :math:`t_0`; the time grid of a simulation must start
        here.
    structure
        Noise structure; overrides the class attribute ``structure``.

    Raises
    ------
    TypeError
        If a dimension is not an integer, `x0` has a wrong type, or the
        subclass does not define a valid :attr:`calculus`.
    ValueError
        If a dimension is not positive, `x0` has a wrong shape or non-finite
        entries, or the structure contradicts the dimensions.
    """

    calculus: ClassVar[Calculus]
    """Calculus in which the equation is interpreted."""

    structure: NoiseStructure = NoiseStructure()
    """Noise structure; after construction, with all verifiable flags added."""

    def __init__(
        self,
        dim: int,
        noise_dim: int,
        x0: Any,
        *,
        t0: float = 0.0,
        structure: NoiseStructure | None = None,
    ) -> None:
        if not isinstance(getattr(type(self), "calculus", None), Calculus):
            msg = (
                f"{type(self).__name__} must define the class attribute "
                "`calculus`; subclass ItoSDE for Itô equations."
            )
            raise TypeError(msg)
        self._dim = positive_int(dim, "dim")
        self._noise_dim = positive_int(noise_dim, "noise_dim")
        self._t0 = finite_float(t0, "t0")
        self._x0, self._x0_is_array = self._validate_x0(x0)
        declared = type(self).structure if structure is None else structure
        self.structure = resolve_structure(declared, self._dim, self._noise_dim)

    def _validate_x0(self, x0: Any) -> tuple[Any, bool]:
        if isinstance(x0, bool):
            msg = "x0 must be a real number, a sequence or an array, got a bool."
            raise TypeError(msg)
        if isinstance(x0, numbers.Real):
            if self._dim != 1:
                msg = (
                    f"A scalar x0 is only allowed for dim == 1, got dim={self._dim}; "
                    f"pass an array of shape ({self._dim},)."
                )
                raise ValueError(msg)
            return finite_float(x0, "x0"), False
        if array_api_compat.is_array_api_obj(x0):
            xp = array_api_compat.array_namespace(x0)
            if not is_real_floating(x0, xp):
                msg = f"x0 must have a real floating-point dtype, got {x0.dtype}."
                raise TypeError(msg)
            self._check_x0_shape(tuple(x0.shape))
            if not bool(xp.all(xp.isfinite(x0))):
                msg = "x0 must contain only finite values."
                raise ValueError(msg)
            return owned_read_only(x0, xp, copy=True), True
        if not isinstance(x0, list | tuple):
            msg = (
                "x0 must be a real number, a sequence of numbers or an array, "
                f"got {x0!r} of type {type(x0).__name__}."
            )
            raise TypeError(msg)
        try:
            array = np.array(x0, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            msg = f"x0 must contain real numbers, got {x0!r}."
            raise TypeError(msg) from exc
        self._check_x0_shape(array.shape)
        if not np.all(np.isfinite(array)):
            msg = "x0 must contain only finite values."
            raise ValueError(msg)
        return owned_read_only(array, np, copy=False), False

    def _check_x0_shape(self, shape: tuple[int, ...]) -> None:
        if shape != (self._dim,):
            msg = f"x0 must have shape ({self._dim},), got {shape}."
            raise ValueError(msg)

    # ------------------------------------------------------------ properties

    @property
    def dim(self) -> int:
        """Dimension :math:`d` of the state."""
        return self._dim

    @property
    def noise_dim(self) -> int:
        """Dimension :math:`m` of the Wiener process."""
        return self._noise_dim

    @property
    def t0(self) -> float:
        """Initial time :math:`t_0`."""
        return self._t0

    @property
    def x0(self) -> Any:
        """Initial value as given: a float (``dim == 1``) or a read-only array."""
        return self._x0

    @property
    def x0_is_array(self) -> bool:
        """Whether `x0` was given as an array, which fixes the backend."""
        return self._x0_is_array

    @property
    def has_exact_solution(self) -> bool:
        """Whether the subclass implements :meth:`exact_solution`."""
        return type(self).exact_solution is not SDEProblem.exact_solution

    # ----------------------------------------------------------- coefficients

    @abstractmethod
    def drift(self, t: float, x: Any) -> Any:
        """Evaluate the drift :math:`f(t, x)`.

        Parameters
        ----------
        t
            Time, a Python float.
        x
            States, an array of shape ``(..., dim)``; the leading axes are
            the batch of paths.

        Returns
        -------
        array
            Shape ``(..., dim)``, in the namespace and on the device of `x`.
        """

    @abstractmethod
    def diffusion(self, t: float, x: Any) -> Any:
        """Evaluate the diffusion :math:`g(t, x)`.

        Parameters
        ----------
        t
            Time, a Python float.
        x
            States, an array of shape ``(..., dim)``.

        Returns
        -------
        array
            Shape ``(..., dim, noise_dim)``, in the namespace and on the
            device of `x`.
        """

    def exact_solution(self, t: Any, W: Any) -> Any:
        r"""Evaluate the exact solution as a function of time and noise.

        Only for problems whose solution :math:`X(t)` is an explicit function
        of :math:`t` and :math:`W(t)`.

        Parameters
        ----------
        t
            Time: a Python float, or an array of shape ``(..., 1)`` that
            broadcasts against `W`.
        W
            Values :math:`W(t) - W(t_0)` of shape ``(..., noise_dim)``.

        Returns
        -------
        array
            :math:`X(t)`, shape ``(..., dim)``.
        """
        msg = f"{type(self).__name__} has no exact solution."
        raise NotImplementedError(msg)

    # -------------------------------------------------------------- helpers

    def initial_value(
        self, n_paths: int, *, xp: ModuleType, dtype: Any, device: Any
    ) -> Any:
        """Return the initial value for a batch of paths.

        Parameters
        ----------
        n_paths
            Number of paths.
        xp, dtype, device
            Namespace, dtype and device of the result.

        Returns
        -------
        array
            A new array of shape ``(n_paths, dim)`` filled with :math:`x_0`.
        """
        n_paths = positive_int(n_paths, "n_paths")
        x0 = self._x0
        if not self._x0_is_array:
            x0 = np.broadcast_to(np.asarray(x0, dtype=np.float64), (self._dim,))
        row = asarray_in(x0, xp, dtype=dtype, device=device)
        return xp.asarray(xp.broadcast_to(row, (n_paths, self._dim)), copy=True)

    def __repr__(self) -> str:
        """Return a summary of the dimensions and the structure."""
        return (
            f"{type(self).__name__}(dim={self._dim}, noise_dim={self._noise_dim}, "
            f"t0={self._t0}, structure={self.structure!r})"
        )


class ItoSDE(SDEProblem):
    r"""Initial value problem of an Itô SDE; subclass it to define a problem.

    See :class:`SDEProblem` for the parameters.

    Examples
    --------
    An Ornstein-Uhlenbeck process
    :math:`\mathrm{d}X(t) = -\lambda X(t)\,\mathrm{d}t + \sigma\,\mathrm{d}W(t)`:

    >>> import numsde as ns
    >>> class OrnsteinUhlenbeck(ns.ItoSDE):
    ...     structure = ns.NoiseStructure(additive=True)
    ...
    ...     def __init__(self, lam=1.0, sigma=0.5, x0=1.0):
    ...         super().__init__(dim=1, noise_dim=1, x0=x0)
    ...         self.lam, self.sigma = lam, sigma
    ...
    ...     def drift(self, t, x):
    ...         return -self.lam * x
    ...
    ...     def diffusion(self, t, x):
    ...         xp = ns.array_namespace(x)
    ...         return self.sigma * xp.ones_like(x)[..., None]
    >>> problem = OrnsteinUhlenbeck()
    >>> sorted(problem.structure.flags)
    ['additive', 'commutative', 'scalar']
    """

    calculus: ClassVar[Calculus] = Calculus.ITO
