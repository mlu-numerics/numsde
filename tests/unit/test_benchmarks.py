import numpy as np
import pytest

from numsde import ItoSDE, NoiseStructure
from numsde.benchmarks import (
    PROBLEMS,
    GeometricBrownianMotion,
    LinearSystem,
    get_problem,
)


def matrix_exponential(M, terms=60):
    result, term = np.eye(M.shape[0]), np.eye(M.shape[0])
    for k in range(1, terms):
        term = term @ M / k
        result = result + term
    return result


# ------------------------------------------------------------------ registry


def test_registry_contains_the_m1_problems():
    assert dict(PROBLEMS) == {
        "geometric_brownian_motion": GeometricBrownianMotion,
        "linear_system": LinearSystem,
    }
    for cls in PROBLEMS.values():
        assert issubclass(cls, ItoSDE)
        assert cls().has_exact_solution
    with pytest.raises(TypeError):
        PROBLEMS["other"] = LinearSystem


def test_get_problem_passes_parameters():
    problem = get_problem("geometric_brownian_motion", mu=2.0)
    assert isinstance(problem, GeometricBrownianMotion)
    assert problem.mu == 2.0


def test_get_problem_rejects_unknown_names():
    with pytest.raises(KeyError, match="registered problems"):
        get_problem("double_well")


# ---------------------------------------------------- geometric Brownian motion


def test_gbm_coefficients_and_structure():
    problem = GeometricBrownianMotion(mu=0.5, sigma=0.3, x0=2.0, t0=1.0)
    x = np.array([[1.0], [4.0]])
    np.testing.assert_allclose(problem.drift(0.0, x), 0.5 * x)
    assert problem.diffusion(0.0, x).shape == (2, 1, 1)
    np.testing.assert_allclose(problem.diffusion(0.0, x)[..., 0], 0.3 * x)
    assert problem.structure == NoiseStructure(
        scalar=True, verified=frozenset({"scalar"})
    )
    assert repr(problem) == (
        "GeometricBrownianMotion(mu=0.5, sigma=0.3, x0=2.0, t0=1.0)"
    )


def test_gbm_exact_solution_broadcasts_over_times():
    problem = GeometricBrownianMotion(mu=0.5, sigma=0.3, x0=2.0, t0=1.0)
    t = np.array([1.0, 1.5])[:, None, None]
    W = np.array([[[0.0]], [[0.4]]])
    X = problem.exact_solution(t, W)
    assert X.shape == (2, 1, 1)
    np.testing.assert_allclose(X[0], 2.0)
    np.testing.assert_allclose(X[1, 0, 0], 2.0 * np.exp((0.5 - 0.045) * 0.5 + 0.12))


@pytest.mark.parametrize("name", ["mu", "sigma", "x0"])
def test_gbm_rejects_non_finite_parameters(name):
    with pytest.raises(ValueError, match=name):
        GeometricBrownianMotion(**{name: np.inf})


# -------------------------------------------------------------- linear system


def test_linear_system_matrices_commute():
    problem = LinearSystem()
    A, B = problem.A, problem.B
    assert B.shape == (2, 2, 2)
    for M in (A, *B):
        for N in (A, *B):
            np.testing.assert_allclose(M @ N, N @ M, atol=1e-15)
    assert not A.flags.writeable
    assert problem.structure.verified == {"commutative"}


def test_linear_system_coefficients():
    problem = LinearSystem()
    x = np.array([[1.0, 2.0], [0.5, -1.0], [0.0, 3.0]])
    np.testing.assert_allclose(problem.drift(0.0, x), x @ problem.A.T)
    g = problem.diffusion(0.0, x)
    assert g.shape == (3, 2, 2)
    for j in range(2):
        np.testing.assert_allclose(g[..., j], x @ problem.B[j].T, atol=1e-15)


def test_linear_system_exact_solution_equals_the_matrix_exponential():
    problem = LinearSystem(t0=0.5)
    A, B = problem.A, problem.B
    t, W = 1.2, np.array([0.3, -0.7])
    M = (A - 0.5 * (B[0] @ B[0] + B[1] @ B[1])) * (t - 0.5) + B[0] * W[0] + B[1] * W[1]
    np.testing.assert_allclose(
        problem.exact_solution(t, W), matrix_exponential(M) @ problem.x0, rtol=1e-13
    )


def test_linear_system_exact_solution_broadcasts_over_times_and_paths():
    problem = LinearSystem()
    t = np.array([0.0, 1.0])[:, None, None]
    W = np.zeros((2, 4, 2))
    X = problem.exact_solution(t, W)
    assert X.shape == (2, 4, 2)
    np.testing.assert_allclose(X[0], np.broadcast_to(problem.x0, (4, 2)))


def test_linear_system_in_one_dimension_is_scalar():
    problem = LinearSystem([[1.0]], [-1.0], [[0.5]], x0=[2.0])
    assert (problem.dim, problem.noise_dim) == (1, 1)
    assert problem.structure.verified == {"scalar", "commutative"}


def test_linear_system_accepts_an_array_initial_value():
    problem = LinearSystem(x0=np.array([1.0, 2.0], dtype=np.float32))
    assert problem.x0_is_array
    assert problem.x0.dtype == np.float32


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"noise_eigenvalues": [0.1, 0.2]}, ValueError, r"\(d, m\) matrix"),
        ({"eigenvectors": [[1.0, 0.0]]}, ValueError, r"eigenvectors must have shape"),
        ({"eigenvectors": [[1.0, 1.0], [1.0, 1.0]]}, ValueError, "invertible"),
        ({"drift_eigenvalues": [1.0]}, ValueError, "drift_eigenvalues"),
        ({"drift_eigenvalues": [1.0, np.nan]}, ValueError, "finite"),
        ({"drift_eigenvalues": ["a", "b"]}, TypeError, "real numbers"),
        ({"x0": [1.0]}, ValueError, r"shape \(2,\)"),
        ({"x0": "ab"}, TypeError, "x0 must be"),
    ],
)
def test_linear_system_rejects_bad_parameters(kwargs, error, match):
    with pytest.raises(error, match=match):
        LinearSystem(**kwargs)
