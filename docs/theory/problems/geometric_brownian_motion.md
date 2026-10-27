# Geometric Brownian motion

Implemented by {class}`numsde.benchmarks.GeometricBrownianMotion`; registry
name `geometric_brownian_motion`.

## Equation

$$
\mathrm{d}X(t) = \mu X(t)\,\mathrm{d}t + \sigma X(t)\,\mathrm{d}W(t),
\qquad X(t_0) = x_0,
$$

with $d = m = 1$.

| Parameter | Meaning | Default |
| --- | --- | --- |
| `mu` | drift $\mu$ | 0.05 |
| `sigma` | volatility $\sigma$ | 0.2 |
| `x0` | initial value $x_0$ | 1.0 |
| `t0` | initial time $t_0$ | 0.0 |

## Noise structure

Scalar ($m = 1$), hence commutative; the library verifies the scalar flag from
the dimension. The noise is multiplicative, not additive.

## Exact solution

By Itô's formula applied to $\log X(t)$,

$$
X(t) = x_0 \exp\Big(\big(\mu - \tfrac{1}{2}\sigma^2\big)(t - t_0)
+ \sigma\big(W(t) - W(t_0)\big)\Big).
$$

The solution is an explicit function of $t$ and $W(t)$, so it is evaluated on
the same Brownian path that drives the scheme; no reference computation is
needed.

## Properties

- Drift and diffusion are linear, hence globally Lipschitz with linear growth:
  all standard convergence results apply.
- Moments: $\mathbb{E}\,X(t) = x_0 e^{\mu (t - t_0)}$ and
  $\mathbb{E}\,X(t)^2 = x_0^2 e^{(2\mu + \sigma^2)(t - t_0)}$.
- Positivity: for $x_0 > 0$ the exact solution stays positive. The
  Euler–Maruyama approximation does not: a step leaves the positive half-line
  whenever $1 + \mu h + \sigma \Delta W_n < 0$, which has positive
  probability for every $h > 0$.
- Stability: $X(t) \to 0$ almost surely if and only if
  $\mu - \tfrac12 \sigma^2 < 0$, and in mean square if and only if
  $2\mu + \sigma^2 < 0$; the two conditions differ, which makes the problem a
  standard example for stability analysis.

## Purpose

The simplest problem with multiplicative noise and a known solution. It is the
first check of every scheme for scalar noise.

| Scheme | Expected strong order | Expected weak order |
| --- | --- | --- |
| Euler–Maruyama | 1/2 | 1 |
| Milstein (M2) | 1 | 1 |

For the test function $\varphi(x) = x$ the Euler–Maruyama approximation has
the exact mean $\mathbb{E}\,X_h^N = x_0 (1 + \mu h)^N$ on a uniform grid, so
its weak error is known in closed form; the statistical tests use this.

## References

{cite}`KloedenPlaten1992`. Registry entry: `numsde.benchmarks.PROBLEMS["geometric_brownian_motion"]`.
