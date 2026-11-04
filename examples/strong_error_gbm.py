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
the same Brownian path.

Unlike the legacy script, which draws new Brownian paths for every step size,
this version samples one path on the finest grid and coarsens it, so all step
sizes see the same realization (see the theory page on Brownian motion). The
estimated errors therefore differ from the legacy ones within the Monte Carlo
error; the regression test ``tests/regression/test_legacy_gbm.py`` checks
that the library reproduces the legacy numbers exactly when it is given the
legacy paths.

Run with the parameters of the book (50 000 paths, about 3 GB of memory)::

    python examples/strong_error_gbm.py

or smaller, with a log-log plot (needs ``numsde[plot]``)::

    python examples/strong_error_gbm.py --n-paths 5000 --plot
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

import numsde as ns


@dataclass(frozen=True)
class ErrorTable:
    """Estimated strong errors with 95 % confidence intervals."""

    n_steps: np.ndarray
    step_sizes: np.ndarray
    errors: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @property
    def eoc(self) -> np.ndarray:
        """Experimental orders of convergence between consecutive step sizes."""
        return np.diff(np.log(self.errors)) / np.diff(np.log(self.step_sizes))

    def format(self) -> str:
        """Return the table as text, in the layout of the legacy output."""
        lines = [
            f"{'N':>6} {'h':>10} {'error':>10} {'EOC':>6}   95% conf. interval",
            "-" * 62,
        ]
        eoc = [float("nan"), *self.eoc]
        for row in zip(
            self.n_steps,
            self.step_sizes,
            self.errors,
            eoc,
            self.lower,
            self.upper,
            strict=True,
        ):
            n, h, e, order, lo, up = row
            order_text = "" if np.isnan(order) else f"{order:6.2f}"
            lines.append(
                f"{n:6d} {h:10.6f} {e:10.6f} {order_text:>6}   [{lo:.6f}, {up:.6f}]"
            )
        return "\n".join(lines)


def strong_errors(
    n_paths: int = 50_000, k_max: int = 8, seed: int = 20250718
) -> ErrorTable:
    """Estimate the strong error of Euler–Maruyama for ``N = 10 * 2**k``.

    Parameters
    ----------
    n_paths
        Number of Monte Carlo paths.
    k_max
        Number of step sizes; ``k = 0, ..., k_max - 1``.
    seed
        Seed of the Brownian path.

    Returns
    -------
    ErrorTable
        Errors and confidence intervals, finest step size last.
    """
    problem = ns.benchmarks.GeometricBrownianMotion(mu=0.25, sigma=1.0, x0=1.0)
    solver = ns.Solver(problem, ns.schemes.EulerMaruyama())
    n_list = np.array([10 * 2**k for k in range(k_max)])
    finest = ns.WienerProcess(1).sample_path(
        ns.uniform_grid(1.0, int(n_list[-1])), n_paths=n_paths, seed=seed
    )
    z = NormalDist().inv_cdf(0.975)

    errors, lower, upper = [], [], []
    for n in n_list:
        path = finest.coarsen(int(n_list[-1] // n))
        x = solver.solve(path).x
        exact = problem.exact_solution(path.times[:, None, None], path.values)
        # Pathwise maximum over the grid of the squared error.
        y = np.max(np.sum((x - exact) ** 2, axis=-1), axis=0)
        mean, half = y.mean(), z * y.std(ddof=1) / np.sqrt(n_paths)
        errors.append(np.sqrt(mean))
        lower.append(np.sqrt(max(mean - half, 0.0)))
        upper.append(np.sqrt(mean + half))
    return ErrorTable(n_list, 1.0 / n_list, *map(np.array, (errors, lower, upper)))


def plot(table: ErrorTable) -> None:
    """Show the errors in a log-log plot with an order line 1/2."""
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    ax.errorbar(
        table.n_steps,
        table.errors,
        yerr=[table.errors - table.lower, table.upper - table.errors],
        ls="none",
        marker="o",
        label="Euler–Maruyama",
    )
    reference = table.errors[0] * (table.n_steps / table.n_steps[0]) ** -0.5
    ax.loglog(table.n_steps, 0.9 * reference, ":", color="gray", label="order 1/2")
    ax.set_xlabel("number of time steps $N$")
    ax.set_ylabel("root-mean-square error, max over the grid")
    ax.legend()
    plt.show()


def main(argv: list[str] | None = None) -> ErrorTable:
    """Run the experiment from the command line and print the table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-paths", type=int, default=50_000)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20250718)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    table = strong_errors(args.n_paths, args.k_max, args.seed)
    print(table.format())
    if args.plot:
        plot(table)
    return table


if __name__ == "__main__":
    main()
