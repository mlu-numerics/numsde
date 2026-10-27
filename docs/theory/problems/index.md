# Benchmark problems

The benchmark suite (`numsde.benchmarks`) collects test problems for
comparing schemes. Every problem has its own page here, analogous to the scheme
pages, with the following sections:

Equation
: The SDE with its parameters, their default values, and the dimensions $d$
  of the state and $m$ of the noise.

Noise structure
: Scalar, additive, diagonal, commutative or general noise.

Exact or reference solution
: The closed-form solution, or how the reference solution is computed and how
  accurate it is.

Properties
: Which standard assumptions hold or fail (global Lipschitz, superlinear
  growth, one-sided Lipschitz), moment bounds, invariants and conserved or
  dissipated quantities.

Purpose
: What the problem is meant to test, with the expected strong and weak orders
  of the relevant schemes and known failure modes.

References
: Origin of the problem in the literature, and a link to the registry entry.

Pages are added together with the problems. Brownian motion on the circle,
the third problem of milestone M1, follows with the convergence study.

```{toctree}
:maxdepth: 1

geometric_brownian_motion
linear_system
```
