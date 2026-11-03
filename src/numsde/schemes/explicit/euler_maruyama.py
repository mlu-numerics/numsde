r"""The Euler–Maruyama scheme.

.. math::

    X_h^{n+1} = X_h^n + f(t_n, X_h^n)\, h_n + g(t_n, X_h^n)\, \Delta W_n .

See the theory page :doc:`/theory/schemes/euler_maruyama`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from numsde.schemes.base import OneStepScheme, matvec

if TYPE_CHECKING:
    from numsde.noise.steps import StepNoise
    from numsde.problems.base import SDEProblem
    from numsde.problems.structure import NoiseStructure

__all__ = ["EulerMaruyama"]


class EulerMaruyama(OneStepScheme):
    r"""Euler–Maruyama scheme for Itô SDEs with general noise.

    The update rule is

    .. math::

        X_h^{n+1} = X_h^n + f(t_n, X_h^n)\, h_n + g(t_n, X_h^n)\, \Delta W_n .

    Under global Lipschitz and linear-growth conditions it converges with
    strong order 1/2 and weak order 1. For additive noise the strong order is
    1, provided the diffusion is Lipschitz continuous in time and the drift
    is smooth in space and Lipschitz continuous in time. The precise
    assumptions are on the theory page.

    Examples
    --------
    >>> import numsde as ns
    >>> problem = ns.benchmarks.GeometricBrownianMotion()
    >>> path = ns.WienerProcess(1).sample_path(
    ...     ns.uniform_grid(1.0, 16), n_paths=4, seed=1
    ... )
    >>> solution = ns.Solver(problem, ns.schemes.EulerMaruyama()).solve(path)
    >>> solution.x.shape
    (17, 4, 1)
    """

    name: ClassVar[str] = "Euler–Maruyama"
    strong_order: ClassVar[float] = 0.5
    weak_order: ClassVar[float | None] = 1.0
    requires: ClassVar[frozenset[str]] = frozenset({"dW"})

    def expected_strong_order(self, structure: NoiseStructure) -> float:
        r"""Return 1 for additive noise and 1/2 otherwise.

        The value is the order expected under the assumptions of the theory
        page. The flag ``additive`` describes only how the diffusion depends
        on the state; it does not certify the time regularity that order 1
        also needs (the diffusion Lipschitz continuous in time, the drift
        smooth in space and Lipschitz continuous in time). For a diffusion
        such as :math:`g(t) = \sqrt{t}` the order is lower although the noise
        is additive.

        Parameters
        ----------
        structure
            Noise structure of the problem.

        Returns
        -------
        float
            The strong order.

        Examples
        --------
        >>> import numsde as ns
        >>> EulerMaruyama().expected_strong_order(ns.NoiseStructure(additive=True))
        1.0
        """
        return 1.0 if structure.additive else self.strong_order

    def step(
        self, problem: SDEProblem, t: float, x: Any, noise: StepNoise, state: Any
    ) -> Any:
        r"""Return :math:`X_h^{n+1} = X_h^n + f h_n + g\,\Delta W_n`.

        See :meth:`Scheme.step <numsde.schemes.Scheme.step>` for the
        parameters.
        """
        f = problem.drift(t, x)
        g = problem.diffusion(t, x)
        return x + f * noise.h + matvec(g, noise.dW)
