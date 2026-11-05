r"""Strong error of the Euler–Maruyama scheme for geometric Brownian motion.

Port of the book project's script ``num_exp_GBM.py`` (Euler–Maruyama part;
the Milstein part follows with milestone M2). The experiment approximates

.. math::

    \mathrm{d}X(t) = \mu X(t)\,\mathrm{d}t + \sigma X(t)\,\mathrm{d}W(t),
    \qquad X(0) = 1, \quad t \in [0, 1],

with :math:`\mu = 1/4`, :math:`\sigma = 1`, and estimates the strong error
uniformly on the grid,

.. math::

    e_h = \Big(\mathbb{E}\Big[\max_n \big|X_h^n - X(t_n)\big|^2\Big]\Big)^{1/2},

for the step counts :math:`N = 10 \cdot 2^k`, against the exact solution on
the same Brownian path, and fits the order of convergence with a bootstrap
confidence interval.

Unlike the legacy script, which draws new Brownian paths for every step size,
this version samples one path on the finest grid and coarsens it, so all step
sizes see the same realization (see the theory page on Brownian motion). The
estimated errors therefore differ from the legacy ones within the Monte Carlo
error; the regression test ``tests/regression/test_legacy_gbm.py`` checks
that the library reproduces the legacy numbers exactly when it is given the
legacy paths.

Run with the parameters of the book (50 000 paths)::

    python examples/strong_error_gbm.py

or smaller, with a log-log plot (needs ``numsde[plot]``)::

    python examples/strong_error_gbm.py --n-paths 5000 --plot
"""

from __future__ import annotations

import argparse

import numsde as ns


def strong_errors(
    n_paths: int = 50_000,
    k_max: int = 8,
    seed: int = 20250718,
    *,
    n_bootstrap: int = 1000,
) -> ns.analysis.StrongConvergenceResult:
    """Estimate the strong error of Euler–Maruyama for ``N = 10 * 2**k``.

    Parameters
    ----------
    n_paths
        Number of Monte Carlo paths.
    k_max
        Number of step sizes; ``k = 0, ..., k_max - 1``.
    seed
        Seed of the Brownian path and of the bootstrap.
    n_bootstrap
        Number of bootstrap replicates for the interval of the order.

    Returns
    -------
    numsde.analysis.StrongConvergenceResult
        Errors, confidence intervals and the fitted order, finest step first.
    """
    problem = ns.benchmarks.GeometricBrownianMotion(mu=0.25, sigma=1.0, x0=1.0)
    n_finest = 10 * 2 ** (k_max - 1)
    path = ns.WienerProcess(1).sample_path(
        ns.uniform_grid(1.0, n_finest), n_paths=n_paths, seed=seed
    )
    study = ns.analysis.StrongConvergenceStudy(
        problem,
        ns.schemes.EulerMaruyama(),
        path,
        factors=[2**k for k in range(k_max)],  # N = n_finest / factor
    )
    return study.run(n_bootstrap=n_bootstrap, seed=seed)


def main(argv: list[str] | None = None) -> ns.analysis.StrongConvergenceResult:
    """Run the experiment from the command line and print the table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-paths", type=int, default=50_000)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20250718)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    result = strong_errors(args.n_paths, args.k_max, args.seed)
    print(result)
    if args.plot:
        import matplotlib.pyplot as plt

        result.plot()
        plt.show()
    return result


if __name__ == "__main__":
    main()
