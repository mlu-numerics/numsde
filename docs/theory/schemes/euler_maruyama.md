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
| Strong, final time, $\big(\mathbb{E}\,\vert X_h^N - X(T)\vert^2\big)^{1/2}$ | $1/2$ | standard | {cite}`KloedenPlaten1992`, Section 10.2 † |
| Strong, uniformly on the grid, $\big(\mathbb{E}\max_n \vert X_h^n - X(t_n)\vert^p\big)^{1/p}$ | $1/2$ | standard, $p \ge 2$ | † reference to be added |
| Strong, additive noise | $1$ | $g$ independent of $x$; $f$ smooth with bounded derivatives | see below |
| Weak, $\vert \mathbb{E}\,\varphi(X_h^N) - \mathbb{E}\,\varphi(X(T))\vert$ | $1$ | $f$, $g$, $\varphi$ smooth with polynomially bounded derivatives | {cite}`KloedenPlaten1992`, Section 14.1 † |

The library reports the strong error uniformly on the grid by default, as
described on the {doc}`theory overview <../index>`.

**Additive noise.** If $g$ does not depend on $x$, all derivatives
$L^i g^j$ vanish and the Euler–Maruyama scheme coincides with the Milstein
scheme. It therefore inherits the strong order 1 of the Milstein scheme. The
method {meth}`~numsde.schemes.EulerMaruyama.expected_strong_order` returns 1
for problems that declare additive noise.

**Failure for superlinear coefficients.** Without the global Lipschitz
condition the scheme may diverge: for drifts that grow faster than linearly,
the moments of $X_h^N$ can blow up as $h \to 0$ even though the exact solution
has bounded moments (†, reference to be added). Tamed, truncated and projected
variants for this case follow in milestone M3.

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

{cite}`KloedenPlaten1992`
