from typing import ClassVar

import numpy as np
import pytest

import numsde as ns
from numsde import NoiseStructure, StepNoise
from numsde.benchmarks import GeometricBrownianMotion, LinearSystem
from numsde.schemes import EulerMaruyama, OneStepScheme, Scheme, matvec


class Identity(OneStepScheme):
    name: ClassVar[str] = "identity"
    strong_order: ClassVar[float] = 0.0

    def step(self, problem, t, x, noise, state):
        return x


class NeedsAdditive(Identity):
    requires_structure: ClassVar[frozenset[str]] = frozenset({"additive"})


class NoCalculus(Identity):
    calculi: ClassVar[frozenset] = frozenset()


def test_matvec_broadcasts_over_batches():
    A = np.arange(12.0).reshape(2, 3, 2)
    v = np.array([[1.0, -1.0], [2.0, 0.5]])
    np.testing.assert_allclose(matvec(A, v), np.einsum("nij,nj->ni", A, v))


def test_scheme_defaults():
    scheme = Identity()
    assert isinstance(scheme, Scheme)
    assert scheme.requires == {"dW"}
    assert scheme.weak_order is None
    assert scheme.expected_strong_order(NoiseStructure(additive=True)) == 0.0
    assert scheme.init_state(GeometricBrownianMotion(), np.ones((1, 1))) is None
    assert repr(scheme) == "Identity()"


def test_check_problem_rejects_unsupported_calculus():
    with pytest.raises(ValueError, match="calculus"):
        NoCalculus().check_problem(GeometricBrownianMotion())


def test_check_problem_rejects_missing_structure():
    with pytest.raises(ValueError, match=r"requires noise structure \['additive'\]"):
        NeedsAdditive().check_problem(GeometricBrownianMotion())


def test_euler_maruyama_attributes():
    em = EulerMaruyama()
    assert em.name == "Euler–Maruyama"
    assert (em.strong_order, em.weak_order) == (0.5, 1.0)
    assert em.requires == {"dW"}
    assert em.expected_strong_order(NoiseStructure(additive=True)) == 1.0
    assert em.expected_strong_order(NoiseStructure(scalar=True)) == 0.5


def test_euler_maruyama_step_by_hand_scalar():
    problem = GeometricBrownianMotion(mu=0.05, sigma=0.2)
    x = np.array([[1.0], [2.0]])
    noise = StepNoise(t=0.0, h=0.5, dW=np.array([[0.3], [-0.1]]))
    x_next = EulerMaruyama().step(problem, 0.0, x, noise, None)
    # x + mu x h + sigma x dW
    np.testing.assert_allclose(x_next, [[1.085], [2.0 + 0.05 - 0.04]])


def test_euler_maruyama_step_multidimensional_noise():
    problem = LinearSystem()
    x = np.array([[1.0, 2.0], [-0.5, 0.25]])
    dW = np.array([[0.1, -0.2], [0.3, 0.05]])
    noise = StepNoise(t=0.0, h=0.01, dW=dW)
    expected = np.stack(
        [
            x[n]
            + problem.A @ x[n] * 0.01
            + sum(problem.B[j] @ x[n] * dW[n, j] for j in range(2))
            for n in range(2)
        ]
    )
    np.testing.assert_allclose(
        EulerMaruyama().step(problem, 0.0, x, noise, None), expected, rtol=1e-14
    )


def test_euler_maruyama_does_not_modify_its_input():
    x = np.ones((2, 1))
    noise = StepNoise(t=0.0, h=0.1, dW=np.ones((2, 1)))
    EulerMaruyama().step(GeometricBrownianMotion(), 0.0, x, noise, None)
    np.testing.assert_array_equal(x, 1.0)


def test_schemes_namespace():
    assert ns.schemes.EulerMaruyama is EulerMaruyama
