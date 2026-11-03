from typing import ClassVar

import numpy as np
import pytest

import numsde as ns
from numsde import ItoSDE, NoiseStructure, SDEProblem


class Additive(ItoSDE):
    """dX = -x dt + dW in d dimensions with m = d."""

    structure = NoiseStructure(additive=True)

    def __init__(self, x0, dim=2, **kwargs):
        super().__init__(dim=dim, noise_dim=dim, x0=x0, **kwargs)

    def drift(self, t, x):
        return -x

    def diffusion(self, t, x):
        xp = ns.array_namespace(x)
        eye = ns.asarray_like(np.eye(self.dim), x)
        return xp.broadcast_to(eye, (*x.shape[:-1], self.dim, self.dim))


def test_problem_properties():
    p = Additive([1.0, 2.0], t0=0.5)
    assert (p.dim, p.noise_dim, p.t0) == (2, 2, 0.5)
    assert p.calculus is ns.Calculus.ITO
    assert not p.x0_is_array
    np.testing.assert_array_equal(p.x0, [1.0, 2.0])
    assert not p.x0.flags.writeable
    assert not p.has_exact_solution
    assert p.structure == NoiseStructure(additive=True)
    assert repr(p) == (
        "Additive(dim=2, noise_dim=2, t0=0.5, "
        "structure=NoiseStructure(additive=True, commutative=True))"
    )


def test_exact_solution_is_not_available_by_default():
    with pytest.raises(NotImplementedError, match="no exact solution"):
        Additive([1.0, 2.0]).exact_solution(1.0, np.zeros(2))


def test_scalar_x0_for_one_dimensional_problem():
    p = Additive(3, dim=1)
    assert p.x0 == 3.0
    assert isinstance(p.x0, float)
    x = p.initial_value(4, xp=ns.array_namespace(), dtype=np.float32, device="cpu")
    assert x.shape == (4, 1)
    assert x.dtype == np.float32
    np.testing.assert_array_equal(x, 3.0)


def test_array_x0_is_copied_and_read_only():
    source = np.array([1.0, 2.0])
    p = Additive(source)
    source[0] = 5.0
    assert p.x0_is_array
    assert p.x0[0] == 1.0
    assert not p.x0.flags.writeable


def test_initial_value_is_a_fresh_writeable_batch():
    p = Additive(np.array([1.0, 2.0], dtype=np.float32))
    x = p.initial_value(3, xp=ns.array_namespace(), dtype=np.float32, device="cpu")
    assert x.shape == (3, 2)
    x[0, 0] = 7.0  # does not touch x0
    assert p.x0[0] == 1.0


def test_structure_argument_overrides_class_attribute():
    p = Additive([1.0, 1.0], structure=NoiseStructure(diagonal=True))
    assert p.structure.diagonal
    assert not p.structure.additive


def test_one_dimensional_noise_is_verified_scalar():
    p = Additive(1.0, dim=1)
    assert p.structure.verified == {"scalar", "commutative"}


@pytest.mark.parametrize(
    ("x0", "error", "match"),
    [
        (1.0, ValueError, "only allowed for dim == 1"),
        ([1.0], ValueError, r"shape \(2,\)"),
        (np.ones(3), ValueError, r"shape \(2,\)"),
        (np.ones((1, 2)), ValueError, r"shape \(2,\)"),
        ([1.0, np.inf], ValueError, "finite"),
        (np.array([1.0, np.nan]), ValueError, "finite"),
        (np.array([1, 2]), TypeError, "floating-point"),
        (True, TypeError, "bool"),
        ("ab", TypeError, "x0 must be"),
        ({1.0, 2.0}, TypeError, "x0 must be"),
        (["a", "b"], TypeError, "real numbers"),
    ],
)
def test_rejects_bad_initial_values(x0, error, match):
    with pytest.raises(error, match=match):
        Additive(x0)


def test_rejects_non_finite_scalar_x0():
    with pytest.raises(ValueError, match="finite"):
        Additive(float("nan"), dim=1)


def test_rejects_bad_dimensions_and_time():
    with pytest.raises(ValueError, match="dim"):
        Additive([1.0], dim=0)
    with pytest.raises(TypeError, match="t0"):
        Additive([1.0, 2.0], t0="0")


def test_subclass_must_define_calculus():
    class NoCalculus(SDEProblem):
        def drift(self, t, x):
            return x

        def diffusion(self, t, x):
            return x[..., None]

    with pytest.raises(TypeError, match="calculus"):
        NoCalculus(dim=1, noise_dim=1, x0=1.0)


def test_subclass_with_invalid_calculus():
    class BadCalculus(SDEProblem):
        calculus: ClassVar = "ito"

        def drift(self, t, x):
            return x

        def diffusion(self, t, x):
            return x[..., None]

    with pytest.raises(TypeError, match="calculus"):
        BadCalculus(dim=1, noise_dim=1, x0=1.0)
