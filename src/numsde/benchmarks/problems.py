r"""Benchmark problems with exact solutions.

Every problem has a theory page under :doc:`/theory/problems/index`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import array_api_compat
import numpy as np

from numsde._backend import array_namespace, asarray_like
from numsde._validation import finite_float
from numsde.problems.base import ItoSDE
from numsde.problems.structure import NoiseStructure
from numsde.schemes.base import matvec

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["GeometricBrownianMotion", "LinearSystem"]


def _matrix(value: Any, shape: tuple[int, ...], name: str) -> npt.NDArray[np.float64]:
    """Return `value` as a read-only float64 array of the given shape."""
    try:
        array = np.array(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must contain real numbers, got {value!r}."
        raise TypeError(msg) from exc
    if array.shape != shape:
        msg = f"{name} must have shape {shape}, got {array.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)
    array.flags.writeable = False
    return array


def _as_list(value: Any) -> Any:
    """Convert nested sequences (e.g. tuples) to the lists ItoSDE accepts."""
    try:
        return np.asarray(value, dtype=np.float64).tolist()
    except (TypeError, ValueError):
        return value


class GeometricBrownianMotion(ItoSDE):
    r"""Geometric Brownian motion, a scalar linear SDE with multiplicative noise.

    The equation is
    :math:`\mathrm{d}X(t) = \mu X(t)\,\mathrm{d}t + \sigma X(t)\,\mathrm{d}W(t)`;
    its exact solution is

    .. math::

        X(t) = x_0 \exp\Big(\big(\mu - \tfrac{1}{2}\sigma^2\big)(t - t_0)
        + \sigma \big(W(t) - W(t_0)\big)\Big).

    See :doc:`/theory/problems/geometric_brownian_motion`.

    Parameters
    ----------
    mu
        Drift parameter :math:`\mu`.
    sigma
        Volatility :math:`\sigma`.
    x0
        Initial value :math:`x_0`.
    t0
        Initial time :math:`t_0`.

    Raises
    ------
    TypeError
        If a parameter is not a real number.
    ValueError
        If a parameter is not finite.

    Examples
    --------
    >>> problem = GeometricBrownianMotion(mu=1.0, sigma=0.5)
    >>> float(problem.exact_solution(1.0, 0.0))
    2.398875293967098
    """

    structure = NoiseStructure(scalar=True)

    def __init__(
        self,
        mu: float = 0.05,
        sigma: float = 0.2,
        x0: float = 1.0,
        *,
        t0: float = 0.0,
    ) -> None:
        super().__init__(dim=1, noise_dim=1, x0=finite_float(x0, "x0"), t0=t0)
        self.mu = finite_float(mu, "mu")
        self.sigma = finite_float(sigma, "sigma")

    def drift(self, t: float, x: Any) -> Any:
        r""":math:`f(t, x) = \mu x`."""
        return self.mu * x

    def diffusion(self, t: float, x: Any) -> Any:
        r""":math:`g(t, x) = \sigma x`, shape ``(..., 1, 1)``."""
        return (self.sigma * x)[..., None]

    def exact_solution(self, t: Any, W: Any) -> Any:
        r""":math:`X(t)` given :math:`W(t) - W(t_0)` of shape ``(..., 1)``."""
        xp = array_namespace(W)
        exponent = (self.mu - 0.5 * self.sigma**2) * (t - self.t0) + self.sigma * W
        return self.x0 * xp.exp(exponent)

    def __repr__(self) -> str:
        """Return the parameters."""
        return (
            f"GeometricBrownianMotion(mu={self.mu}, sigma={self.sigma}, "
            f"x0={self.x0}, t0={self.t0})"
        )


class LinearSystem(ItoSDE):
    r"""Linear system with commuting matrices and multidimensional noise.

    .. math::

        \mathrm{d}X(t) = A X(t)\,\mathrm{d}t
        + \sum_{j=1}^m B_j X(t)\,\mathrm{d}W^j(t),

    with :math:`A = Q \operatorname{diag}(a) Q^{-1}` and
    :math:`B_j = Q \operatorname{diag}(b_{\cdot j}) Q^{-1}`. The matrices
    commute, so the noise is commutative and the exact solution is

    .. math::

        X(t) = \exp\Big(\big(A - \tfrac12 \textstyle\sum_j B_j^2\big)(t - t_0)
        + \sum_j B_j \big(W^j(t) - W^j(t_0)\big)\Big)\, x_0 .

    See :doc:`/theory/problems/linear_system`.

    Parameters
    ----------
    eigenvectors
        Invertible matrix :math:`Q` of shape ``(d, d)``.
    drift_eigenvalues
        Eigenvalues :math:`a \in \mathbb{R}^d` of :math:`A`.
    noise_eigenvalues
        Matrix :math:`b \in \mathbb{R}^{d \times m}`; column :math:`j` holds
        the eigenvalues of :math:`B_j`.
    x0
        Initial value :math:`x_0 \in \mathbb{R}^d`.
    t0
        Initial time :math:`t_0`.

    Raises
    ------
    TypeError
        If a parameter does not contain real numbers.
    ValueError
        If shapes do not match, values are not finite or :math:`Q` is
        numerically singular.

    Examples
    --------
    >>> problem = LinearSystem()
    >>> problem.dim, problem.noise_dim
    (2, 2)
    >>> sorted(problem.structure.verified)
    ['commutative']
    """

    def __init__(
        self,
        eigenvectors: Any = ((1.0, 1.0), (0.0, 1.0)),
        drift_eigenvalues: Any = (-1.0, -0.5),
        noise_eigenvalues: Any = ((0.3, 0.1), (0.2, 0.4)),
        x0: Any = (1.0, 1.0),
        *,
        t0: float = 0.0,
    ) -> None:
        b = np.array(noise_eigenvalues, dtype=np.float64)
        if b.ndim != 2:
            msg = f"noise_eigenvalues must be a (d, m) matrix, got shape {b.shape}."
            raise ValueError(msg)
        d, m = b.shape
        Q = _matrix(eigenvectors, (d, d), "eigenvectors")
        if np.linalg.cond(Q) > 1e10:
            msg = "eigenvectors must be an invertible, well-conditioned matrix."
            raise ValueError(msg)
        Q_inv = np.linalg.inv(Q)
        self.eigenvectors = Q
        self.drift_eigenvalues = _matrix(drift_eigenvalues, (d,), "drift_eigenvalues")
        self.noise_eigenvalues = _matrix(b, (d, m), "noise_eigenvalues")
        a, b = self.drift_eigenvalues, self.noise_eigenvalues
        self.A = _matrix(Q @ np.diag(a) @ Q_inv, (d, d), "A")
        self.B = _matrix(
            np.stack([Q @ np.diag(b[:, j]) @ Q_inv for j in range(m)]),
            (m, d, d),
            "B",
        )
        # g(x)[k, j] = (B_j x)[k], as one (d*m, d) matrix applied to x.
        self._G = np.ascontiguousarray(np.moveaxis(self.B, 0, 1)).reshape(d * m, d)
        self._Q_inv = Q_inv
        super().__init__(
            dim=d,
            noise_dim=m,
            x0=x0 if array_api_compat.is_array_api_obj(x0) else _as_list(x0),
            t0=t0,
            # The matrices commute by construction, so this is verified.
            structure=NoiseStructure(
                commutative=True, verified=frozenset({"commutative"})
            ),
        )

    def drift(self, t: float, x: Any) -> Any:
        r""":math:`f(t, x) = A x`."""
        return matvec(asarray_like(self.A, x), x)

    def diffusion(self, t: float, x: Any) -> Any:
        r""":math:`g(t, x) = (B_1 x, \dots, B_m x)`, shape ``(..., d, m)``."""
        xp = array_namespace(x)
        g = matvec(asarray_like(self._G, x), x)
        return xp.reshape(g, (*tuple(x.shape[:-1]), self.dim, self.noise_dim))

    def exact_solution(self, t: Any, W: Any) -> Any:
        r""":math:`X(t)` given :math:`W(t) - W(t_0)` of shape ``(..., m)``.

        Computed in the eigenbasis, where every component is a geometric
        Brownian motion.
        """
        xp = array_namespace(W)
        a, b = self.drift_eigenvalues, self.noise_eigenvalues
        c = asarray_like(a - 0.5 * np.sum(b**2, axis=1), W)
        y0 = matvec(asarray_like(self._Q_inv, W), asarray_like(self.x0, W))
        exponent = (t - self.t0) * c + matvec(asarray_like(b, W), W)
        return matvec(asarray_like(self.eigenvectors, W), xp.exp(exponent) * y0)
