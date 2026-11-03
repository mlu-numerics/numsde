# Euler–Maruyama scheme

The Euler–Maruyama scheme is the stochastic counterpart of the explicit Euler
method and the reference point for every other scheme in the library. It is
implemented by {class}`numsde.schemes.EulerMaruyama`.

## Update rule

On a grid $t_0 < t_1 < \dots < t_N$ with step sizes $h_n = t_{n+1} - t_n$
and Brownian increments $\Delta W_n = W(t_{n+1}) - W(t_n)$,

$$
X_h^{n+1} = X_h^n + f(t_n, X_h^n)\, h_n + g(t_n, X_h^n)\, \Delta W_n,
\qquad X_h^0 = x_0 .
$$

Here $g(t_n, X_h^n)\,\Delta W_n = \sum_{j=1}^m g^j(t_n, X_h^n)\,\Delta W_n^j$
is a matrix–vector product, so the scheme applies to general noise without
any further random quantities. In the code, `step()` reads exactly like the
formula:

```python
f = problem.drift(t, x)
g = problem.diffusion(t, x)
return x + f * noise.h + matvec(g, noise.dW)
```

## Derivation

Integrating the SDE over one step gives

$$
X(t_{n+1}) = X(t_n) + \int_{t_n}^{t_{n+1}} f(s, X(s))\,\mathrm{d}s
+ \int_{t_n}^{t_{n+1}} g(s, X(s))\,\mathrm{d}W(s).
$$

Freezing both integrands at the left end point $s = t_n$ yields the scheme.
Evaluating $g$ at the *left* end point is essential: it makes the discrete
stochastic integral an Itô sum, so the scheme is consistent with Itô
calculus. The neglected terms are the first terms of the Itô–Taylor expansion;
the largest of them, $\sum_{i,j} L^i g^j \, I_{ij}$, is of size $h_n$ in mean
square and limits the strong order to $1/2$. Adding it gives the Milstein
scheme {cite}`KloedenPlaten1992`.

## Convergence

Let $h = \max_n h_n$. The orders below hold under the standard assumptions:
$f$ and $g$ are globally Lipschitz continuous in $x$, satisfy a linear growth
bound, and are Hölder continuous of order $1/2$ in $t$; $x_0$ is
deterministic.

| Error | Order | Assumptions | Reference |
| --- | --- | --- | --- |
| Strong, final time in $L^1$, $\mathbb{E}\,\vert X_h^N - X(T)\vert$ | $1/2$ | standard | {cite}`KloedenPlaten1992`, Theorem 10.2.2 |
| Strong, final time in $L^2$, $\big(\mathbb{E}\,\vert X_h^N - X(T)\vert^2\big)^{1/2}$ | $1/2$ | standard | {cite}`MilsteinTretyakov2021`, Theorem 1.1.1 (fundamental theorem on mean-square order of convergence) |
| Strong, uniformly on the grid, $\big(\mathbb{E}\max_n \vert X_h^n - X(t_n)\vert^2\big)^{1/2}$ | $1/2$ | standard | {cite}`Mao2008`, Chapter 2, Theorem 7.3 |
| Strong, uniformly on the grid, $\big(\mathbb{E}\max_n \vert X_h^n - X(t_n)\vert^p\big)^{1/p}$ | $1/2$ | standard, $p \ge 2$ | Kanagawa (1988), *Yokohama Math. J.* 36, 79–86 † |
| Strong, additive noise | $1$ | $g = g(t)$ independent of $x$ and Lipschitz continuous in $t$; $f$ twice continuously differentiable in $x$ with bounded first and second derivatives, and Lipschitz continuous in $t$ | † see below |
| Weak, $\vert \mathbb{E}\,\varphi(X_h^N) - \mathbb{E}\,\varphi(X(T))\vert$ | $1$ | $f$, $g$, $\varphi$ smooth with polynomially bounded derivatives | {cite}`KloedenPlaten1992`, Section 14.1 |

† Not yet checked against the source. Kanagawa (1988) is said to contain the
proof for $p \ge 2$; the article could not be consulted so far, so it is not
yet in the bibliography.

The $L^1$ and $L^2$ errors at the final time are bounded by the $L^2$ error
uniformly on the grid, so order $1/2$ of the latter implies order $1/2$ of
both. The fundamental theorem bounds $\max_n \big(\mathbb{E}\,\vert X_h^n -
X(t_n)\vert^2\big)^{1/2}$, with the maximum outside the expectation.

The library reports the strong error uniformly on the grid by default, as
described on the {doc}`theory overview <../index>`.

**Additive noise.** If $g$ does not depend on $x$, all derivatives
$L^i g^j$ vanish and the Euler–Maruyama scheme coincides with the Milstein
scheme. This removes the error term that limits the order to $1/2$ in
general, but strong order 1 needs more than additivity: the coefficients must
also be regular enough *in time*, as stated in the table. Freezing $g$ at the
left end point leaves the error
$\sum_n \int_{t_n}^{t_{n+1}} \big(g(s) - g(t_n)\big)\,\mathrm{d}W(s)$. The
terms are centered and independent, so by the Itô isometry the error is of
order $h$ if $g$ is Lipschitz continuous in $t$; Hölder continuity of order
$1/2$, which suffices for the general order $1/2$, does not suffice here.

*Example.* For $\mathrm{d}X(t) = \sqrt{t}\,\mathrm{d}W(t)$, $X(0) = 0$, on
$[0, 1]$ with $h = 1/N$, the noise is additive, the drift vanishes and $g$ is
Hölder continuous of order $1/2$, but not Lipschitz continuous at $t = 0$. The
Itô isometry gives

$$
\mathbb{E}\,\vert X(1) - X_h^N \vert^2
= \sum_{n=0}^{N-1} \int_{nh}^{(n+1)h} \big(\sqrt{t} - \sqrt{nh}\big)^2\,\mathrm{d}t
\asymp h^2 \log(1/h),
$$

because the term with index $n \ge 1$ is about $h^2/(12 n)$. Even the
final-time error in $L^2$ is therefore not $O(h)$.

The flags of a {class}`~numsde.NoiseStructure` describe how $g$ depends on
$x$; they certify nothing about regularity in $t$. Accordingly,
{meth}`~numsde.schemes.EulerMaruyama.expected_strong_order` returns 1 for a
problem that declares additive noise *on the assumption* that the time
regularity of the table holds; for the time-independent benchmark problems it
does.

**Failure for superlinear coefficients.** Without the global Lipschitz
condition the scheme may diverge although the exact solution has finite
moments. For a scalar SDE
$\mathrm{d}X(t) = f(X(t))\,\mathrm{d}t + g(X(t))\,\mathrm{d}W(t)$ on a uniform
grid, {cite}`HutzenthalerJentzenKloeden2010`, Theorem 2.1, shows: if one
coefficient grows superlinearly and the other more slowly, that is, for some
$C > 0$, $\beta > 1$ and $0 \le \alpha < \beta$ and all $\vert x\vert \ge C$,

$$
\vert f(x)\vert \ge \frac{\vert x\vert^\beta}{C}
\quad\text{and}\quad \vert g(x)\vert \le C \vert x\vert^\alpha,
$$

or the same with $f$ and $g$ exchanged, and if $g(X(0)) \ne 0$ with positive
probability, then for every $p \in [1, \infty)$ with
$\mathbb{E}\,\vert X(T)\vert^p < \infty$

$$
\lim_{N \to \infty} \mathbb{E}\,\vert X(T) - X_h^N\vert^p = \infty
\quad\text{and}\quad
\lim_{N \to \infty} \big\vert \mathbb{E}\,\vert X(T)\vert^p
- \mathbb{E}\,\vert X_h^N\vert^p \big\vert = \infty .
$$

The scheme thus diverges in the strong and in the numerically weak sense. The
reason is a sequence of events of at least exponentially small probability on
which the approximation grows at least double-exponentially fast. The
standard example is $\mathrm{d}X(t) = -X(t)^3\,\mathrm{d}t + \mathrm{d}W(t)$,
whose solution has finite moments of every order. Tamed, truncated and
projected variants for this case follow in milestone M3.

## Tests

- Unit tests compare single steps with values computed by hand for scalar and
  multidimensional noise.
- Property-based tests check that the scheme reduces to the explicit Euler
  method without noise, reproduces $x_0 + W(t_n)$ exactly for
  $\mathrm{d}X = \mathrm{d}W$, and that every path of a batch is advanced
  independently.
- Statistical tests (marker `slow`) fit the strong order on
  {doc}`geometric Brownian motion <../problems/geometric_brownian_motion>` and
  the {doc}`linear system <../problems/linear_system>`, and check the weak
  error of $\mathbb{E}\,X_h^N$ for geometric Brownian motion against its exact
  value.

## References

{cite}`KloedenPlaten1992`, {cite}`MilsteinTretyakov2021`, {cite}`Mao2008`,
{cite}`HutzenthalerJentzenKloeden2010`; Kanagawa (1988), not yet checked.
