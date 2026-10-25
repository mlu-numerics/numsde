# Theory

Every scheme in NumSDE has a theory page with

- the update rule, in the notation used by the implementation,
- a sketch of its derivation,
- its strong and weak convergence orders and the assumptions behind them,
- references, and
- a link to the implementing class.

## Notation

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

Theory pages for individual schemes are added together with their
implementation.

```{toctree}
:maxdepth: 1

brownian_motion
```
