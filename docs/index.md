# NumSDE

**Numerical methods for stochastic differential equations, for teaching and research.**

NumSDE provides readable, thoroughly tested reference implementations of numerical
schemes for Itô stochastic differential equations

$$
\mathrm{d}X(t) = f(t, X(t))\,\mathrm{d}t + g(t, X(t))\,\mathrm{d}W(t),
\qquad X(t) \in \mathbb{R}^d,\; W(t) \in \mathbb{R}^m,
$$

together with tools for empirical convergence studies.

:::{note}
NumSDE is in early development (milestone M0, foundations). The planned scope is
described in the {doc}`architecture and roadmap <development/architecture>`.
:::

```{toctree}
:maxdepth: 2
:caption: User guide

tutorials/index
howto/index
theory/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
errata
references
```

```{toctree}
:maxdepth: 2
:caption: Development

development/index
```
