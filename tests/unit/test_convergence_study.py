import sys
import types

import numpy as np
import pytest

import numsde as ns
from numsde import ItoSDE, NoiseStructure, Solver, WienerProcess
from numsde.analysis import StrongConvergenceResult, StrongConvergenceStudy
from numsde.benchmarks import GeometricBrownianMotion, LinearSystem
from numsde.schemes import EulerMaruyama
from numsde.stepping import uniform_grid

EM = EulerMaruyama()


def gbm_path(n_steps=2**6, n_paths=300, seed=1):
    return WienerProcess(1).sample_path(
        uniform_grid(1.0, n_steps), n_paths=n_paths, seed=seed
    )


def gbm():
    return GeometricBrownianMotion(mu=1.0, sigma=1.0)


def direct_errors(problem, path, factors, error="grid", p=2.0):
    """Per-path data computed with solve() and stored trajectories."""
    rows = []
    for k in factors:
        coarse = path.coarsen(k)
        x = Solver(problem, EM).solve(coarse).x
        exact = problem.exact_solution(coarse.times[:, None, None], coarse.values)
        err = np.sum((x - exact) ** 2, axis=-1) ** (p / 2)
        rows.append(err.max(axis=0) if error == "grid" else err[-1])
    return np.stack(rows)


class NoExactSolution(ItoSDE):
    structure = NoiseStructure(additive=True)

    def __init__(self):
        super().__init__(dim=1, noise_dim=1, x0=0.0)

    def drift(self, t, x):
        return -x

    def diffusion(self, t, x):
        return ns.array_namespace(x).ones_like(x)[..., None]


class PureNoise(ItoSDE):
    """dX = dW: Euler-Maruyama is exact."""

    structure = NoiseStructure(additive=True)

    def __init__(self):
        super().__init__(dim=1, noise_dim=1, x0=0.0)

    def drift(self, t, x):
        return 0.0 * x

    def diffusion(self, t, x):
        return ns.array_namespace(x).ones_like(x)[..., None]

    def exact_solution(self, t, W):
        return W


# ------------------------------------------------------------- simulation


@pytest.mark.parametrize("error", ["grid", "final"])
def test_pathwise_data_equal_the_errors_of_stored_solutions(error):
    path = gbm_path()
    study = StrongConvergenceStudy(gbm(), EM, path, [1, 2, 8], error=error)
    np.testing.assert_allclose(
        study.simulate()[0], direct_errors(gbm(), path, [1, 2, 8], error), rtol=1e-12
    )


def test_exponent_p():
    path = gbm_path()
    study = StrongConvergenceStudy(gbm(), EM, path, [2, 4], p=4)
    np.testing.assert_allclose(
        study.simulate()[0], direct_errors(gbm(), path, [2, 4], p=4.0), rtol=1e-12
    )


def test_factors_are_sorted_and_give_step_sizes():
    study = StrongConvergenceStudy(gbm(), EM, gbm_path(), [8, 2, 4])
    assert study.factors == (2, 4, 8)
    np.testing.assert_allclose(study.step_sizes, [2 / 64, 4 / 64, 8 / 64])
    assert study.problem.mu == 1.0
    assert study.schemes == (EM,)


def test_step_size_of_a_non_uniform_grid_is_the_largest_step():
    times = np.array([0.0, 0.1, 0.5, 0.6, 1.0])
    path = WienerProcess(1).sample_path(times, n_paths=5, seed=0)
    study = StrongConvergenceStudy(gbm(), EM, path, [1, 2])
    np.testing.assert_allclose(study.step_sizes, [0.4, 0.5])


def test_several_schemes():
    result = StrongConvergenceStudy(gbm(), [EM, EM], gbm_path(), [1, 2, 4]).run(
        n_bootstrap=50, seed=0
    )
    assert result.scheme_names == ("Euler–Maruyama", "Euler–Maruyama")
    np.testing.assert_array_equal(result.pathwise[0], result.pathwise[1])
    np.testing.assert_array_equal(result.orders_ci[0], result.orders_ci[1])


def test_multidimensional_problem():
    problem = LinearSystem()
    path = WienerProcess(2).sample_path(uniform_grid(1.0, 32), n_paths=50, seed=2)
    result = StrongConvergenceStudy(problem, EM, path, [1, 2, 4]).run(
        n_bootstrap=50, seed=0
    )
    assert result.errors.shape == (1, 3)
    assert np.all(np.diff(result.errors[0]) > 0)


# -------------------------------------------------------------- reference


@pytest.mark.parametrize("error", ["grid", "final"])
def test_scheme_reference_on_a_fine_grid_approaches_the_exact_reference(error):
    path = gbm_path(n_paths=400)
    exact = StrongConvergenceStudy(gbm(), EM, path, [2, 4, 8], error=error)
    fine = StrongConvergenceStudy(
        gbm(), EM, path, [2, 4, 8], error=error, reference=EM, reference_refinement=32
    )
    e_exact = np.sqrt(exact.simulate().mean(-1))
    e_fine = np.sqrt(fine.simulate().mean(-1))
    np.testing.assert_allclose(e_fine, e_exact, rtol=0.05)


def test_scheme_reference_without_refinement_uses_the_path_itself():
    path = gbm_path()
    study = StrongConvergenceStudy(gbm(), EM, path, [2, 4], reference=EM)
    reference = Solver(gbm(), EM).solve(path).x
    x = Solver(gbm(), EM).solve(path.coarsen(4)).x
    expected = np.max(np.sum((x - reference[::4]) ** 2, axis=-1), axis=0)
    np.testing.assert_allclose(study.simulate()[0, 1], expected, rtol=1e-12)


def test_reference_description():
    path = gbm_path()
    assert "exact solution" in repr(StrongConvergenceStudy(gbm(), EM, path, [1, 2]))
    study = StrongConvergenceStudy(
        gbm(), EM, path, [1, 2], reference=EM, reference_refinement=4
    )
    assert "Euler–Maruyama with 256 steps" in repr(study)


def test_problem_without_exact_solution_needs_a_scheme_reference():
    path = gbm_path()
    with pytest.raises(ValueError, match="no exact solution"):
        StrongConvergenceStudy(NoExactSolution(), EM, path, [1, 2])
    result = StrongConvergenceStudy(
        NoExactSolution(), EM, path, [2, 4, 8], reference=EM
    ).run(n_bootstrap=50, seed=0)
    assert np.all(np.isfinite(result.orders))


# --------------------------------------------------------------- analysis


def test_run_analyzes_the_simulated_data():
    path = gbm_path()
    study = StrongConvergenceStudy(gbm(), EM, path, [1, 2, 4, 8])
    result = study.run(n_bootstrap=100, seed=3)
    y = direct_errors(gbm(), path, [1, 2, 4, 8])
    np.testing.assert_allclose(result.errors[0], np.sqrt(y.mean(-1)), rtol=1e-12)
    expected = np.polyfit(np.log(study.step_sizes), np.log(result.errors[0]), 1)[0]
    assert result.orders[0] == pytest.approx(expected)
    lower, upper = result.orders_ci[0]
    assert lower < upper
    assert result.expected_orders.tolist() == [0.5]
    assert result.n_paths == 300
    assert result.method == "percentile"
    assert (result.p, result.error, result.reference) == (2.0, "grid", "exact solution")
    assert np.all(result.errors_ci[0, :, 0] <= result.errors[0])
    assert np.all(result.errors[0] <= result.errors_ci[0, :, 1])
    assert not result.errors.flags.writeable
    assert result.eoc.shape == (1, 3)


def test_bootstrap_is_reproducible_with_a_seed():
    study = StrongConvergenceStudy(gbm(), EM, gbm_path(), [1, 2, 4])
    a = study.run(n_bootstrap=100, seed=7)
    b = study.run(n_bootstrap=100, seed=7)
    np.testing.assert_array_equal(a.orders_ci, b.orders_ci)


def test_refit_excludes_step_sizes_and_keeps_the_seed():
    result = StrongConvergenceStudy(gbm(), EM, gbm_path(), [1, 2, 4, 8]).run(
        n_bootstrap=100, seed=1
    )
    np.testing.assert_array_equal(result.refit().orders_ci, result.orders_ci)
    partial = result.refit(fit_factors=[1, 2, 4])
    assert partial.fit_mask.tolist() == [True, True, True, False]
    expected = np.polyfit(
        np.log(result.step_sizes[:3]), np.log(result.errors[0, :3]), 1
    )
    assert partial.orders[0] == pytest.approx(expected[0])
    np.testing.assert_array_equal(partial.errors, result.errors)
    assert "(not in fit)" in partial.format()
    bca = result.refit(method="bca", n_bootstrap=50, confidence=0.9, seed=2)
    assert (bca.method, bca.n_bootstrap, bca.confidence) == ("bca", 50, 0.9)


def test_refit_rejects_unknown_factors():
    result = StrongConvergenceStudy(gbm(), EM, gbm_path(), [1, 2, 4]).run(
        n_bootstrap=20, seed=0
    )
    with pytest.raises(ValueError, match=r"fit_factors \[3\]"):
        result.refit(fit_factors=[1, 3])
    with pytest.raises(ValueError, match="two distinct step sizes"):
        result.refit(fit_factors=[1])


def test_exact_scheme_has_undefined_order():
    path = WienerProcess(1).sample_path(uniform_grid(1.0, 16), n_paths=20, seed=0)
    result = StrongConvergenceStudy(PureNoise(), EM, path, [1, 2, 4]).run(
        n_bootstrap=20, seed=0
    )
    np.testing.assert_allclose(result.errors, 0.0, atol=1e-15)
    assert np.isnan(result.orders[0])
    assert np.all(np.isnan(result.orders_ci))
    assert "nan" in result.format()


def test_format_lists_every_step_size():
    result = StrongConvergenceStudy(
        gbm(), EM, gbm_path(), [1, 2, 4], error="final"
    ).run(n_bootstrap=20, seed=0, method="bca", confidence=0.9)
    text = str(result)
    assert text.startswith("Strong error at the final time, p = 2")
    assert "Euler–Maruyama (expected order 0.5)" in text
    assert "90% CI" in text
    assert "(BCa, 20 resamples)" in text
    assert len(text.splitlines()) == 1 + 1 + 2 + 3 + 1


def test_from_pathwise_validates_its_input():
    kwargs = {
        "step_sizes": [0.1, 0.05],
        "factors": [2, 1],
        "scheme_names": ["s"],
        "expected_orders": [0.5],
        "p": 2.0,
        "error": "grid",
        "reference": "exact",
        "n_bootstrap": 10,
    }
    y = np.ones((1, 2, 5))
    with pytest.raises(ValueError, match="pathwise must have shape"):
        StrongConvergenceResult.from_pathwise(np.ones((2, 5)), **kwargs)
    with pytest.raises(ValueError, match="do not match"):
        StrongConvergenceResult.from_pathwise(y, **{**kwargs, "factors": [1]})
    with pytest.raises(ValueError, match="method must be one of"):
        StrongConvergenceResult.from_pathwise(y, **kwargs, method="basic")


# ------------------------------------------------------------------- plot


class _Recorder:
    """Stands in for a matplotlib Axes object and records the calls."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append(name)

        return record


def test_plot_draws_errors_and_fitted_lines(monkeypatch):
    ax = _Recorder()
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.subplots = lambda: (None, ax)
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.pyplot = pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    result = StrongConvergenceStudy(gbm(), [EM, EM], gbm_path(), [1, 2, 4]).run(
        n_bootstrap=20, seed=0
    )
    assert result.plot() is ax
    assert ax.calls.count("errorbar") == 2
    assert ax.calls.count("plot") == 2
    other = _Recorder()
    assert result.plot(ax=other) is other


def test_plot_with_matplotlib():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    result = StrongConvergenceStudy(gbm(), EM, gbm_path(), [1, 2, 4]).run(
        n_bootstrap=20, seed=0
    )
    ax = result.plot()
    assert ax.get_xscale() == "log"
    assert len(ax.get_lines()) >= 1


# ------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"problem": "gbm"}, TypeError, "problem must be an instance of SDEProblem"),
        ({"schemes": []}, ValueError, "At least one scheme"),
        ({"schemes": "EM"}, TypeError, "schemes must be a Scheme"),
        ({"schemes": [EM, "EM"]}, TypeError, "Scheme instances"),
        ({"path": np.zeros((3, 1, 1))}, TypeError, "BrownianPath"),
        ({"factors": 2}, TypeError, "factors must be a sequence"),
        ({"factors": [2]}, ValueError, "two distinct"),
        ({"factors": [2, 2, 4]}, ValueError, "two distinct"),
        ({"factors": [2, 3]}, ValueError, "does not divide"),
        ({"factors": [True, 2]}, TypeError, "factors"),
        ({"p": 0.5}, ValueError, "at least 1"),
        ({"p": float("inf")}, ValueError, "at least 1"),
        ({"p": True}, TypeError, "p must be a real number"),
        ({"error": "max"}, ValueError, "error must be one of"),
        ({"reference": "fine"}, TypeError, "reference must be"),
        ({"reference_refinement": 2}, ValueError, "only to a scheme reference"),
        ({"reference": EM, "factors": [1, 2]}, ValueError, "finer than every grid"),
        ({"reference_refinement": 0}, ValueError, "reference_refinement"),
    ],
)
def test_study_rejects_bad_arguments(kwargs, error, match):
    args = {"problem": gbm(), "schemes": EM, "path": gbm_path(), "factors": [1, 2]}
    args.update(kwargs)
    problem, schemes, path, factors = (
        args.pop(k) for k in ("problem", "schemes", "path", "factors")
    )
    with pytest.raises(error, match=match):
        StrongConvergenceStudy(problem, schemes, path, factors, **args)


def test_study_checks_the_path_against_the_problem_at_once():
    path = WienerProcess(2).sample_path(uniform_grid(1.0, 4), n_paths=3, seed=0)
    with pytest.raises(ValueError, match="noise_dim=1"):
        StrongConvergenceStudy(gbm(), EM, path, [1, 2])


# Regression tests for the review of PR #16.


@pytest.mark.parametrize(
    ("refinement", "error"), [(1, "grid"), (2, "grid"), (1, "final")]
)
def test_scheme_reference_on_non_nested_grids(refinement, error):
    # Finding P1: with the factors 2 and 3 on 12 steps the coarse grids are
    # not nested; the reference must be taken at the same times.
    problem = GeometricBrownianMotion(mu=1.0, sigma=0.0)
    path = WienerProcess(1).sample_path(uniform_grid(1.0, 12), n_paths=3, seed=0)
    study = StrongConvergenceStudy(
        problem,
        EM,
        path,
        [2, 3],
        reference=EM,
        reference_refinement=refinement,
        error=error,
    )
    reference = Solver(problem, EM).solve(path.refine(refinement)).x[::refinement]
    for row, factor in zip(study.simulate()[0], (2, 3), strict=True):
        x = Solver(problem, EM).solve(path.coarsen(factor)).x
        err = np.sum((x - reference[::factor]) ** 2, axis=-1)
        expected = err.max(axis=0) if error == "grid" else err[-1]
        np.testing.assert_allclose(row, expected, rtol=1e-12)
