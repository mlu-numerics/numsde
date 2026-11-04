"""Regression test against the legacy script ``num_exp_GBM.py`` of the book project.

The reference numbers in ``data/legacy_gbm_em_l2error.npz`` were produced by
the unmodified legacy function ``compute_L2_errors`` with the Euler–Maruyama
solver only, at a reduced sample size (see ``data/README.md``). The test feeds
the legacy Brownian paths into the library and requires the same errors.

Legacy sampling convention (``StochODEs.Wiener.simulate``): one generator
``numpy.random.default_rng(seed)`` serves all step sizes in the order of
``n_list``; for each step size it draws the increments with shape
``(n_paths, m, n)`` and sums them cumulatively along time. These are other
random numbers than :meth:`numsde.WienerProcess.sample_path` draws for the
same seed, so the paths are rebuilt here and wrapped as
:class:`numsde.BrownianPath`.

The legacy error is the default strong error of the library, uniformly on the
grid with p = 2. On the machine that generated the data the results agree
bitwise; the tolerance covers platform-dependent rounding of ``exp``.
"""

from pathlib import Path

import numpy as np
import pytest

import numsde as ns

DATA = Path(__file__).parent / "data" / "legacy_gbm_em_l2error.npz"


@pytest.fixture(scope="module")
def reference():
    with np.load(DATA, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def legacy_brownian_paths(seed, n_list, n_paths, t0, t_end):
    """Yield the Brownian paths of the legacy script, one per step size."""
    rng = np.random.default_rng(int(seed))
    for n in n_list:
        t = np.linspace(t0, t_end, int(n) + 1)
        dW = rng.normal(loc=0.0, scale=np.sqrt(np.diff(t)), size=(int(n_paths), 1, n))
        W = np.zeros((int(n_paths), 1, int(n) + 1))
        W[:, :, 1:] = np.cumsum(dW, axis=2)
        # Legacy layout (n_paths, m, n + 1) -> library layout (n + 1, n_paths, m).
        yield ns.BrownianPath(t, np.moveaxis(W, 2, 0), seed=0)


def strong_error_on_grid(problem, path, x):
    """(E[max_n |X_h^n - X(t_n)|^2])^(1/2), computed as in the legacy script."""
    exact = problem.exact_solution(path.times[:, None, None], path.values)
    return np.sqrt(np.mean(np.max(np.sum((exact - x) ** 2, axis=-1), axis=0)))


def test_reference_data_documents_its_origin(reference):
    assert str(reference["script"]).startswith("num_exp_GBM.py")
    assert int(reference["seed"]) == 20250718
    assert len(str(reference["sha256_num_exp_GBM"])) == 64
    assert reference["errors"].shape == reference["n_list"].shape


def test_euler_maruyama_reproduces_the_legacy_errors(reference):
    problem = ns.benchmarks.GeometricBrownianMotion(
        mu=float(reference["mu"]),
        sigma=float(reference["sigma"]),
        x0=float(reference["x0"]),
        t0=float(reference["t0"]),
    )
    solver = ns.Solver(problem, ns.schemes.EulerMaruyama())
    paths = legacy_brownian_paths(
        reference["seed"],
        reference["n_list"],
        reference["n_paths"],
        float(reference["t0"]),
        float(reference["t_end"]),
    )
    errors = [strong_error_on_grid(problem, p, solver.solve(p).x) for p in paths]
    np.testing.assert_allclose(errors, reference["errors"], rtol=1e-12, atol=0)
