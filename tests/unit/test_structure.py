import pytest

from numsde import Calculus, NoiseStructure
from numsde.problems import FLAGS, resolve_structure


def test_empty_structure_has_no_flags():
    s = NoiseStructure()
    assert s.flags == frozenset()
    assert s.verified == frozenset()
    assert repr(s) == "NoiseStructure()"


@pytest.mark.parametrize("flag", ["scalar", "additive", "diagonal"])
def test_special_noise_implies_commutative(flag):
    s = NoiseStructure(**{flag: True})
    assert s.commutative
    assert s.flags == {flag, "commutative"}
    assert s.declared_only == s.flags


def test_verified_flags_propagate_implications():
    s = NoiseStructure(scalar=True, verified=frozenset({"scalar"}))
    assert s.verified == {"scalar", "commutative"}
    assert s.declared_only == frozenset()


def test_with_verified_sets_and_verifies():
    s = NoiseStructure(additive=True).with_verified("scalar")
    assert s.flags == {"scalar", "additive", "commutative"}
    assert s.verified == {"scalar", "commutative"}
    assert s.declared_only == {"additive"}


def test_repr_lists_flags_and_verified():
    s = NoiseStructure(additive=True).with_verified("scalar")
    assert repr(s) == (
        "NoiseStructure(scalar=True, additive=True, commutative=True, "
        "verified={'scalar', 'commutative'})"
    )
    assert repr(NoiseStructure(verified=frozenset())) == "NoiseStructure()"


def test_structure_is_frozen():
    s = NoiseStructure()
    with pytest.raises(AttributeError):
        s.scalar = True


def test_rejects_non_bool_flag():
    with pytest.raises(TypeError, match="additive must be a bool"):
        NoiseStructure(additive=1)


def test_rejects_unknown_verified_flag():
    with pytest.raises(ValueError, match="Unknown noise-structure flag"):
        NoiseStructure(verified=frozenset({"triangular"}))


def test_rejects_non_string_verified_flag():
    with pytest.raises(TypeError, match="flag names"):
        NoiseStructure(verified=frozenset({1}))


def test_cannot_verify_unset_flag():
    with pytest.raises(ValueError, match="not set"):
        NoiseStructure(verified=frozenset({"additive"}))


def test_with_verified_rejects_unknown_flag():
    with pytest.raises(ValueError, match="Unknown"):
        NoiseStructure().with_verified("upper")


def test_flag_names():
    assert FLAGS == ("scalar", "additive", "diagonal", "commutative")
    assert Calculus.ITO.value == "ito"


# ------------------------------------------------------------ resolve_structure


def test_resolve_marks_one_dimensional_noise_as_scalar():
    s = resolve_structure(NoiseStructure(), dim=3, noise_dim=1)
    assert s.verified == {"scalar", "commutative"}


def test_resolve_keeps_multidimensional_structure():
    s = NoiseStructure(diagonal=True)
    assert resolve_structure(s, dim=2, noise_dim=2) == s


def test_resolve_rejects_scalar_with_several_noise_dimensions():
    with pytest.raises(ValueError, match="Scalar noise requires noise_dim == 1"):
        resolve_structure(NoiseStructure(scalar=True), dim=1, noise_dim=2)


def test_resolve_rejects_diagonal_with_unequal_dimensions():
    with pytest.raises(ValueError, match="dim == noise_dim"):
        resolve_structure(NoiseStructure(diagonal=True), dim=3, noise_dim=2)


def test_resolve_rejects_non_structure():
    with pytest.raises(TypeError, match="NoiseStructure"):
        resolve_structure("scalar", dim=1, noise_dim=1)
