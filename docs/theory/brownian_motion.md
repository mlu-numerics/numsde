# Brownian motion and the Brownian bridge

Every simulation in NumSDE is driven by a sampled Brownian path. This page
describes how paths are sampled, refined and coarsened, and why these
operations keep all comparisons between step sizes consistent. The
implementation is {class}`numsde.BrownianPath`, created by
{meth}`numsde.WienerProcess.sample_path`.

## Sampling on a grid

An $m$-dimensional standard Brownian motion $W$ has independent Gaussian
increments: for $s < t$,

$$
W(t) - W(s) \sim \mathcal{N}\big(0, (t - s)\, I_m\big).
$$

On a grid $t_0 < t_1 < \dots < t_N$ with step sizes $h_n = t_{n+1} - t_n$ the
path is therefore sampled exactly by

$$
W(t_0) = 0, \qquad
W(t_{n+1}) = W(t_n) + \sqrt{h_n}\, Z_n, \qquad Z_n \sim \mathcal{N}(0, I_m)
\ \text{i.i.d.}
$$

No discretization error is involved: the sampled values have exactly the joint
law of Brownian motion at the grid points {cite}`KloedenPlaten1992`.

## Refinement with the Brownian bridge

Convergence studies, adaptive step sizes and multilevel Monte Carlo need the
*same* Brownian path at a finer resolution. A new point $t \in (s, u)$ between
two grid points must then be sampled conditionally on the values already known.
Given $W(s) = a$ and $W(u) = b$, the value $W(t)$ is Gaussian with

$$
\mathbb{E}\big[W(t) \mid W(s) = a,\ W(u) = b\big]
= a + \frac{t - s}{u - s}\,(b - a),
\qquad
\operatorname{Var}\big[W(t) \mid \cdot\big] = \frac{(t - s)(u - t)}{u - s}\, I_m .
$$

This is the *Brownian bridge*. By the Markov property, values outside $[s, u]$
carry no further information, so the formula above is the full conditional
law. Several new points in one interval are sampled from left to right, each
conditioned on its left neighbor, which may itself be new, and on $W(u)$.
Points after the last grid point $t_N$ extend the path with independent
increments.

Refining therefore never changes values that were already sampled. In
particular, for every integer $k$,

```python
path.refine(k).coarsen(k)  # reproduces `path` exactly
```

## Coarsening

Coarsening by a factor $k$ keeps every $k$-th grid point. The coarse increments
are then the sums of $k$ consecutive fine increments,

$$
W(t_{(j+1)k}) - W(t_{jk}) = \sum_{i=0}^{k-1} \Delta W_{jk+i},
$$

so a scheme on the coarse grid and a scheme on the fine grid are driven by the
same Brownian motion. This is the basis of every strong convergence study in
NumSDE: a path is sampled once on the finest grid, and all coarser
approximations are computed from its coarsenings.

## A path is a fixed realization

A {class}`~numsde.BrownianPath` behaves like a value, not like a simulation
that is still running:

- **Its values never change.** The path owns a private copy of its values, and
  every operation returns a new path that shares no memory with the old one.
  With NumPy the values and the time grid live in immutable memory, so they
  cannot be made writeable again. PyTorch has no read-only tensors; there
  NumSDE only guarantees that no two paths share memory, and the values must
  not be modified in place.
- **Operations are deterministic.** A path carries an immutable seed instead of
  a random number generator. `insert` and `refine` derive their random numbers
  from that seed and the requested times, so `path.refine(2)` returns the same
  path on every call, independent of what was done with the path before.

The second property is weaker than it may sound. Two different requests, for
example `path.refine(4)` and `path.refine(2).refine(2)`, are both exact samples
from the conditional law, but they are different samples: the values at the
common grid points need not agree. Paths that answer every query consistently,
in any order, need a different construction (a Brownian tree); it is planned
for adaptive step sizes in milestone M3.

## Reproducibility across backends

Random numbers are drawn on the host with NumPy's PCG64 generator and then
converted to the requested backend. A path sampled with the same seed is
therefore identical on NumPy and on PyTorch, on the CPU and on a GPU, up to the
chosen floating-point precision.

## Example

```python
from numsde import WienerProcess, uniform_grid

W = WienerProcess(dim=2)
fine = W.sample_path(uniform_grid(1.0, 2**10), n_paths=10_000, seed=42)

# The same Brownian motion on the grids with step sizes 2^-9, ..., 2^-4:
coarse = [fine.coarsen(2**j) for j in range(1, 7)]
```
