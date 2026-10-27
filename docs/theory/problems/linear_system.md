# Linear system with commuting matrices

Implemented by {class}`numsde.benchmarks.LinearSystem`; registry name
`linear_system`.

## Equation

$$
\mathrm{d}X(t) = A X(t)\,\mathrm{d}t + \sum_{j=1}^m B_j X(t)\,\mathrm{d}W^j(t),
\qquad X(t_0) = x_0 \in \mathbb{R}^d .
$$

The matrices are built from a common eigenbasis $Q$, so that they commute:

$$
A = Q \operatorname{diag}(a)\, Q^{-1}, \qquad
B_j = Q \operatorname{diag}(b_{1j}, \dots, b_{dj})\, Q^{-1}.
$$

| Parameter | Meaning | Default |
| --- | --- | --- |
| `eigenvectors` | $Q \in \mathbb{R}^{d \times d}$, invertible | $\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$ |
| `drift_eigenvalues` | $a \in \mathbb{R}^d$ | $(-1, -0.5)$ |
| `noise_eigenvalues` | $b \in \mathbb{R}^{d \times m}$, column $j$ for $B_j$ | $\begin{pmatrix} 0.3 & 0.1 \\ 0.2 & 0.4 \end{pmatrix}$ |
| `x0` | initial value | $(1, 1)$ |
| `t0` | initial time | 0 |

With the defaults, $d = m = 2$ and

$$
A = \begin{pmatrix} -1 & 0.5 \\ 0 & -0.5 \end{pmatrix}, \quad
B_1 = \begin{pmatrix} 0.3 & -0.1 \\ 0 & 0.2 \end{pmatrix}, \quad
B_2 = \begin{pmatrix} 0.1 & 0.3 \\ 0 & 0.4 \end{pmatrix}.
$$

## Noise structure

The $j$-th column of the diffusion is $g^j(x) = B_j x$, so
$L^i g^j(x) = B_j B_i x$. Because the matrices commute, the noise is
**commutative**; this holds by construction and is recorded as verified. In
general it is neither additive nor diagonal, and scalar only for $m = 1$.

## Exact solution

For commuting matrices the solution is

$$
X(t) = \exp\Big(\big(A - \tfrac12 \textstyle\sum_j B_j^2\big)(t - t_0)
+ \sum_j B_j \big(W^j(t) - W^j(t_0)\big)\Big)\, x_0 .
$$

In the coordinates $Y = Q^{-1} X$ the system decouples into $d$ geometric
Brownian motions driven by the same $m$-dimensional Wiener process,

$$
Y_i(t) = Y_i(t_0) \exp\Big(\big(a_i - \tfrac12 \textstyle\sum_j b_{ij}^2\big)(t - t_0)
+ \sum_j b_{ij} \big(W^j(t) - W^j(t_0)\big)\Big),
$$

which is how the implementation evaluates it; no matrix exponential and no
SciPy are needed. The tests compare it with a truncated exponential series.

## Properties

- Linear coefficients: globally Lipschitz with linear growth.
- Mean: $\mathbb{E}\,X(t) = e^{A (t - t_0)} x_0$, because the Itô integrals
  have mean zero.
- With the default parameters all eigenvalues of $A$ are negative, so the mean
  decays; the components in the eigenbasis behave like the geometric Brownian
  motion of the {doc}`geometric_brownian_motion` page.

## Purpose

The first problem with multidimensional state and noise. It tests the general
matrix–vector code path of the schemes and the vectorization over paths, with
an exact solution on the same Brownian path. Because the noise is
commutative, the Milstein scheme reaches strong order 1 without iterated
integrals (Lévy areas).

| Scheme | Expected strong order | Expected weak order |
| --- | --- | --- |
| Euler–Maruyama | 1/2 | 1 |
| Milstein for commutative noise (M2) | 1 | 1 |

A variant with non-commuting matrices, which needs Lévy areas for order 1 and
has no closed-form solution, is planned for milestone M2.

## References

{cite}`KloedenPlaten1992`. Registry entry: `numsde.benchmarks.PROBLEMS["linear_system"]`.
