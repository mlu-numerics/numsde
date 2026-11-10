---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Getting started: from an SDE to its order of convergence

This tutorial walks through the core workflow of NumSDE in four steps:

1. define an SDE and simulate paths with the Euler–Maruyama scheme,
2. drive several step sizes with one realization of the Brownian motion,
3. measure the strong error against an exact solution in a convergence study,
4. measure it against a fine reference solution when no exact solution is
   available.

It runs as part of the documentation build, so every number and figure below
was produced by the current code. To run it yourself, install NumSDE with
plotting support (`pip install "numsde[plot]"`) and open this file in Jupyter
with [Jupytext](https://jupytext.readthedocs.io), or copy the cells.

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np

import numsde as ns
```

## 1. Define an SDE and simulate it

An SDE is an initial value problem

$$
\mathrm{d}X(t) = f(t, X(t))\,\mathrm{d}t + g(t, X(t))\,\mathrm{d}W(t),
\qquad X(t_0) = x_0,
$$

with drift $f(t, x) \in \mathbb{R}^d$ and diffusion
$g(t, x) \in \mathbb{R}^{d \times m}$. You define one by subclassing
{class}`numsde.ItoSDE` and implementing `drift` and `diffusion`. Both receive
a whole batch of states $x$ of shape `(n_paths, d)` at once; the diffusion
returns shape `(n_paths, d, m)`.

As a first example take the Ornstein–Uhlenbeck process

$$
\mathrm{d}X(t) = -\lambda X(t)\,\mathrm{d}t + \sigma\,\mathrm{d}W(t),
$$

with $d = m = 1$. Its diffusion does not depend on $x$, so the noise is
*additive*, which we declare with a {class}`numsde.NoiseStructure`:

```{code-cell} python
class OrnsteinUhlenbeck(ns.ItoSDE):
    """dX = -lam X dt + sigma dW."""

    structure = ns.NoiseStructure(additive=True)

    def __init__(self, lam=2.0, sigma=0.5, x0=1.0):
        super().__init__(dim=1, noise_dim=1, x0=x0)
        self.lam, self.sigma = lam, sigma

    def drift(self, t, x):
        return -self.lam * x

    def diffusion(self, t, x):
        xp = ns.array_namespace(x)  # works for NumPy and PyTorch arrays
        return self.sigma * xp.ones_like(x)[..., None]


problem = OrnsteinUhlenbeck()
problem
```

The noise is a {class}`numsde.BrownianPath`: one realization of the Wiener
process on a time grid, for a batch of independent paths. Sampling it is
separate from solving, and the seed makes it reproducible.

```{code-cell} python
path = ns.WienerProcess(1).sample_path(
    ns.uniform_grid(1.0, 2**8), n_paths=2000, seed=3
)
path.values.shape  # (grid points, paths, noise dimension)
```

A {class}`numsde.Solver` combines a problem with a scheme and runs the time
loop for all paths at once.

```{code-cell} python
solver = ns.Solver(problem, ns.schemes.EulerMaruyama())
solution = solver.solve(path)

fig, ax = plt.subplots()
ax.plot(solution.t, solution.x[:, :20, 0], lw=0.8)
ax.set_xlabel("$t$")
ax.set_ylabel("$X_h(t)$")
ax.set_title("Euler–Maruyama, 20 of 2000 paths")
plt.show()
```

The mean and variance of the Ornstein–Uhlenbeck process are known:
$\mathbb{E}\,X(t) = e^{-\lambda t} x_0$ and
$\operatorname{Var} X(t) = \frac{\sigma^2}{2\lambda}(1 - e^{-2\lambda t})$.
The sample estimates at $t = 1$ agree up to the Monte Carlo error and the
small bias of the scheme:

```{code-cell} python
x_final = solution.final[:, 0]
lam, sigma = problem.lam, problem.sigma
print(f"mean:     {x_final.mean():.4f}  (exact {np.exp(-lam):.4f})")
var_exact = sigma**2 / (2 * lam) * (1 - np.exp(-2 * lam))
print(f"variance: {x_final.var():.4f}  (exact {var_exact:.4f})")
```

## 2. One realization for all step sizes

Convergence is about what happens to the error as the step size $h$ shrinks
*for the same noise*. {meth}`numsde.BrownianPath.coarsen` turns a path into
the same realization on a coarser grid: the coarse increments are sums of the
fine ones. Solving on coarser and coarser versions of one path shows the
approximations approaching each other:

```{code-cell} python
fig, ax = plt.subplots()
for factor in (32, 8, 1):
    coarse = path.coarsen(factor)
    x = solver.solve(coarse).x
    ax.plot(coarse.times, x[:, 0, 0], marker="." if factor > 1 else None,
            label=f"h = {coarse.step_sizes[0]:.4f}")
ax.set_xlabel("$t$")
ax.set_ylabel("$X_h(t)$, first path")
ax.legend()
plt.show()
```

This coupling is what makes error estimates meaningful: with independent
noise for each step size, the differences between the curves would be
dominated by the noise, not by the scheme.

## 3. A convergence study against the exact solution

Geometric Brownian motion
$\mathrm{d}X = \mu X\,\mathrm{d}t + \sigma X\,\mathrm{d}W$ has the exact
solution $X(t) = x_0 \exp\big((\mu - \tfrac12\sigma^2)t + \sigma W(t)\big)$,
a function of $W(t)$ on the same path. It is available as a benchmark
problem, see {doc}`../theory/problems/geometric_brownian_motion`.

A {class}`numsde.analysis.StrongConvergenceStudy` runs the scheme on
`path.coarsen(k)` for every factor `k` and estimates the strong error
uniformly on the grid,

$$
e_h = \Big(\mathbb{E}\Big[\max_n \big|X_h^n - X(t_n)\big|^2\Big]\Big)^{1/2}.
$$

```{code-cell} python
gbm = ns.benchmarks.GeometricBrownianMotion(mu=1.0, sigma=1.0)
study = ns.analysis.StrongConvergenceStudy(
    gbm,
    ns.schemes.EulerMaruyama(),
    path,
    factors=[1, 2, 4, 8, 16, 32],
    reference="exact",
)
result = study.run(seed=0)
print(result)
```

The table lists, per step size, the estimated error with a 95 % confidence
interval and the experimental order of convergence (EOC) between neighboring
step sizes. Below it, the order fitted by least squares on a log-log scale,
with a bootstrap confidence interval: the paths are resampled with the same
indices for all step sizes and the fit is repeated. The expected order 1/2
comes from the theory of the scheme, see
{doc}`../theory/schemes/euler_maruyama`.

```{code-cell} python
ax = result.plot()
ax.set_title("Euler–Maruyama for geometric Brownian motion")
plt.show()
```

If the coarsest step sizes are not yet in the asymptotic range, you can leave
them out of the fit without simulating again:

```{code-cell} python
refitted = result.refit(fit_factors=[1, 2, 4, 8])
print(f"all step sizes:      {result.orders[0]:.3f}, CI {result.orders_ci[0].round(3)}")
print(f"four finest only:    {refitted.orders[0]:.3f}, CI {refitted.orders_ci[0].round(3)}")
```

How reliable these intervals are, and why they are only approximate when the
pathwise errors have heavy tails, is explained on the theory page
{doc}`../theory/convergence_studies`.

## 4. A convergence study against a reference solution

The Ornstein–Uhlenbeck solution
$X(t) = e^{-\lambda t} x_0 + \sigma \int_0^t e^{-\lambda(t-s)}\,\mathrm{d}W(s)$
depends on the whole path of $W$, not only on $W(t)$, so there is no
`exact_solution` to compare with. Instead, the study can use a scheme on a
finer grid as reference: the path is refined by the Brownian bridge, here
with 16 substeps per step, so the reference sees the same noise as all coarse
approximations.

```{code-cell} python
study = ns.analysis.StrongConvergenceStudy(
    problem,
    ns.schemes.EulerMaruyama(),
    path,
    factors=[1, 2, 4, 8, 16, 32],
    reference=ns.schemes.EulerMaruyama(),
    reference_refinement=16,
)
result = study.run(seed=0)
print(result)
```

For additive noise, Euler–Maruyama coincides with the Milstein scheme and
converges with strong order 1 instead of 1/2, provided the coefficients are
regular enough in time. The study reports this expected order because the
problem declares additive noise, and the fitted order agrees.

## Where to go next

- {doc}`../theory/index`: the theory behind schemes, benchmark problems and
  convergence studies.
- {doc}`../api/index`: all classes and functions with runnable examples.
- `examples/strong_error_gbm.py` in the repository: the convergence
  experiment of the companion book, ported to NumSDE.
