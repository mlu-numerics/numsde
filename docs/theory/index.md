# Theory

The theory pages explain the mathematics behind the library: one page per
scheme and one page per benchmark problem.

Every scheme in NumSDE has a theory page with

- the update rule, in the notation used by the implementation,
- a sketch of its derivation,
- its strong and weak convergence orders and the assumptions behind them,
- references, and
- a link to the implementing class.

## Notation

Stochastic processes are written with their time argument in parentheses,
$X(t)$ and $W(t)$, never as $X_t$. Subscripts are reserved for discretizations:
$X_h$ is a numerical approximation with step size $h$ and $X_h^n$ its value at
the grid point $t_n$, so that $X_h$ cannot be mistaken for $X(h)$. Increments
over a step carry subscripts too, for example
$\Delta W_n = W(t_{n+1}) - W(t_n)$; components of a vector-valued process are
indexed by superscripts, as in $W^j(t)$.

Throughout, $f\colon [0,T]\times\mathbb{R}^d \to \mathbb{R}^d$ denotes the drift,
$g\colon [0,T]\times\mathbb{R}^d \to \mathbb{R}^{d\times m}$ the diffusion, and
$W$ an $m$-dimensional Wiener process. A numerical approximation on the grid
$0 = t_0 < t_1 < \dots < t_N = T$ with maximal step size $h$ is written
$X_h^n \approx X(t_n)$.

The default strong error used throughout the library is the error uniformly on
the grid,

$$
e_h^{\mathrm{strong}} = \Big(\mathbb{E}\Big[\max_{n \in \{0,\dots,N\}}
\big|X_h^n - X(t_n)\big|^p\Big]\Big)^{1/p}.
$$

Theory pages for individual schemes (under `schemes/`) and benchmark problems
(under `problems/`) are added together with their implementation.

```{toctree}
:maxdepth: 1

brownian_motion
problems/index
```
