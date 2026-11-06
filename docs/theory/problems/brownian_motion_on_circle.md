# Brownian motion on a circle

Implemented by {class}`numsde.benchmarks.BrownianMotionOnCircle`; registry
name `brownian_motion_on_circle`.

## Equation

$$
\mathrm{d}X(t) = -\tfrac12 \sigma^2 X(t)\,\mathrm{d}t + \sigma J X(t)\,\mathrm{d}W(t),
\qquad X(t_0) = x_0 \in \mathbb{R}^2,
\qquad J = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix},
$$

with $d = 2$ and $m = 1$. Componentwise,
$\mathrm{d}X^1 = -\tfrac12\sigma^2 X^1\,\mathrm{d}t - \sigma X^2\,\mathrm{d}W$ and
$\mathrm{d}X^2 = -\tfrac12\sigma^2 X^2\,\mathrm{d}t + \sigma X^1\,\mathrm{d}W$.

| Parameter | Meaning | Default |
| --- | --- | --- |
| `sigma` | angular diffusion $\sigma$ | 1.0 |
| `x0` | initial value $x_0$ | $(1, 0)$ |
| `t0` | initial time $t_0$ | 0.0 |

## Noise structure

Scalar ($m = 1$), hence commutative; the library verifies the scalar flag from
the dimension. The noise is multiplicative: $g(x) = \sigma J x$.

## Exact solution

Write $x_0 = r(\cos\theta_0, \sin\theta_0)$ and
$\varphi(t) = \theta_0 + \sigma (W(t) - W(t_0))$. Itô's formula gives
$\mathrm{d}\cos\varphi = -\sin\varphi\,\sigma\,\mathrm{d}W
- \tfrac12 \sigma^2 \cos\varphi\,\mathrm{d}t$ and the analogous formula for
$\sin\varphi$, so $X(t) = r(\cos\varphi(t), \sin\varphi(t))$ solves the
equation. Equivalently,

$$
X(t) = R\big(\sigma (W(t) - W(t_0))\big)\, x_0, \qquad
R(\varphi) = \begin{pmatrix} \cos\varphi & -\sin\varphi \\
\sin\varphi & \cos\varphi \end{pmatrix}.
$$

The drift matrix $-\tfrac12\sigma^2 I$ and the diffusion matrix $\sigma J$
commute, and $(\sigma J)^2 = -\sigma^2 I$, so this is also the formula
$\exp\big((A - \tfrac12 B^2)(t - t_0) + B (W(t) - W(t_0))\big)\,x_0$ of the
{doc}`linear system <linear_system>`; the tests check both. The solution is
a function of $W(t)$ alone and is evaluated on the same Brownian path that
drives the scheme.

## Properties

- Linear coefficients: globally Lipschitz with linear growth, so all standard
  convergence results apply.
- **Invariant:** $|X(t)| = |x_0|$ for all $t$; the solution moves on the
  circle of radius $|x_0|$.
- Mean: $\mathbb{E}\,X(t) = e^{-\sigma^2 (t - t_0)/2}\, x_0$, which decays to
  zero although every path keeps its radius: the angle spreads over the
  circle.
- **Euler–Maruyama leaves the circle.** Because $J x \perp x$ and
  $|J x| = |x|$, one step gives exactly

  $$
  |X_h^{n+1}|^2 = |X_h^n|^2 \Big(\big(1 - \tfrac12\sigma^2 h_n\big)^2
  + \sigma^2 (\Delta W_n)^2\Big).
  $$

  The factors are independent with mean $1 + \sigma^4 h_n^2/4$, so on a
  uniform grid with $N$ steps of size $h$

  $$
  \mathbb{E}\,|X_h^N|^2 = |x_0|^2 \big(1 + \tfrac14 \sigma^4 h^2\big)^N
  \approx |x_0|^2 \big(1 + \tfrac14 \sigma^4 h\,(T - t_0)\big):
  $$

  in the mean square the approximation drifts outwards by $O(h)$, while each
  single path may move inwards or outwards.

## Purpose

The simplest problem with a geometric invariant. It tests the
multidimensional code path with scalar noise and shows that standard schemes
do not preserve invariants, which motivates the structure-preserving schemes
of milestone M4.

| Scheme | Expected strong order | Expected weak order |
| --- | --- | --- |
| Euler–Maruyama | 1/2 | 1 |
| Milstein (M2) | 1 | 1 |

The statistical tests check the strong order of Euler–Maruyama and the exact
second moment $\mathbb{E}\,|X_h^N|^2$ above; a property-based test checks the
radius identity of every step on every path.

## References

Registry entry: `numsde.benchmarks.PROBLEMS["brownian_motion_on_circle"]`.
A literature reference for the problem is still to be added.
