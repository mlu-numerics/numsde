# Architecture and roadmap

This page is the authoritative description of NumSDE's scope, architecture and
roadmap. Changes go through pull requests like any code change; decisions are
recorded in the [decision log](#decision-log) at the end.

## Vision and scope

NumSDE is a BSD-3-licensed Python library of numerical methods for Itô SDEs,
built for university teaching and research. A teaching release must be ready by
1 September 2027, ahead of the next lecture on the topic.

$$
\mathrm{d}X(t) = f(t, X(t))\,\mathrm{d}t + g(t, X(t))\,\mathrm{d}W(t),
\qquad X(t) \in \mathbb{R}^d,\; W(t) \in \mathbb{R}^m,\; g(t,x) \in \mathbb{R}^{d \times m}
$$

**Goals**

- Readable, verified reference implementations of explicit one-step and
  multistep schemes for multidimensional SDEs with general noise.
- Tools for empirical strong and weak convergence studies, plus a benchmark
  suite to test new schemes against established ones from the literature.
- Didactic documentation: a theory page for every scheme and every benchmark
  problem, and executable tutorials.
- High reliability through layered testing and full coverage.

**In scope**

- Itô calculus; finite-dimensional systems driven by an $m$-dimensional Wiener
  process.
- General noise, with specialized fast paths for scalar, additive, diagonal and
  commutative noise.
- Non-stiff problems: explicit one-step and multistep schemes.
- Superlinearly growing coefficients: tamed, truncated and projected schemes;
  adaptive step sizes for growth control in the sense of Kelly and Lord.
- Second-order SODEs (e.g. the stochastic pendulum), including symplectic and
  splitting schemes.
- Strong convergence first, weak convergence second; multilevel Monte Carlo
  (MLMC).
- Vectorized simulation of many paths; NumPy backend first, PyTorch (GPU)
  through the same abstraction.
- Derivatives supplied by the user or generated symbolically with SymPy.

**Designed for, implemented later**

- Stratonovich calculus, fractional Brownian motion, Lévy noise, rough paths.
- Finite-difference and autodiff derivative providers.
- Implicit schemes for stiff problems.

**Out of scope**

- SPDEs (candidate for a follow-up project) and delay SDEs.
- Adaptive step sizes for error control.

The library stays independent of the book project with W.-J. Beyn; the two may
share a bibliography, simulation results and theory.

## Design principles

Every design decision is judged first by whether a student can read the code
next to the formula, and second by performance.

1. **Code reads like the formula.** One class per scheme; its `step()` method
   mirrors the textbook update rule, with the same symbol names where possible.
2. **Separation of concerns.** Problem, noise, scheme, step control and
   analysis are independent objects that meet only in the solver.
3. **Noise is a first-class object.** A Brownian path can be stored, refined,
   coarsened and reused across schemes and step sizes. This is what makes
   convergence studies, adaptive stepping and MLMC correct.
4. **Vectorized by default.** Every scheme operates on a batch of paths; a
   single path is a batch of size one.
5. **Backend-agnostic core.** Numerical code is written against the Python
   Array API standard, so NumPy and PyTorch run the same scheme code.
6. **Open for extension.** Stratonovich calculus, fractional Brownian motion or
   Lévy noise arrive as new problem or noise types; schemes declare what they
   support instead of being rewritten.
7. **Lean core.** Only NumPy and `array-api-compat` are hard dependencies;
   further libraries (e.g. SciPy) are added only when a module needs them; SymPy,
   matplotlib and PyTorch are optional extras.
8. **Nothing ships unverified.** Each scheme comes with unit, property and
   convergence tests, a docstring with a runnable example, and a theory page;
   each benchmark problem comes with tests of its exact solution or reference
   and a theory page.

## Package layout

```text
numsde/
├── src/numsde/
│   ├── _backend/          # array namespace resolution, RNG wrapper
│   ├── problems/          # SDEProblem, ItoSDE, NoiseStructure, SecondOrderSDE,
│   │                      # SymbolicSDE, derivative providers
│   ├── noise/             # WienerProcess, BrownianPath, refinement, coarsening
│   ├── integrals/         # iterated stochastic integrals and Lévy areas
│   ├── schemes/
│   │   ├── base.py        # Scheme, OneStepScheme, MultiStepScheme
│   │   ├── explicit/      # Euler–Maruyama, Milstein, Wagner–Platen, SRK
│   │   ├── nonlinear/     # tamed, truncated, projected schemes
│   │   ├── multistep/     # linear multistep schemes, e.g. BDF2-Maruyama
│   │   └── geometric/     # symplectic and splitting schemes
│   ├── stepping/          # time grids, FixedStep, adaptive growth control
│   ├── solver.py          # Solver: orchestrates problem, scheme, noise, stepping
│   ├── solution.py        # Solution container
│   ├── analysis/          # strong / weak error, order estimation, plots
│   ├── mlmc/              # multilevel Monte Carlo estimators
│   ├── benchmarks/        # problem library and benchmark runner
│   └── plotting.py        # optional matplotlib helpers
├── tests/                 # unit/, property/, statistical/, regression/
├── docs/                  # Sphinx + MyST
├── examples/              # ported legacy scripts as runnable examples
└── .github/workflows/
```

## Core abstractions

The solver is the only place where the abstractions meet:

```text
                    ┌────────────────────────────┐
                    │ Analysis                   │
                    │ convergence, MLMC, bench.  │
                    └─────────────┬──────────────┘
                                  │ calls
┌──────────────────┐   ┌──────────▼─────────┐   ┌──────────────────┐
│ Problem          │──▶│ Solver             │──▶│ Solution         │
│ drift, diffusion │   │ time loop, batch   │   │ times, paths     │
└──────────────────┘   └──┬───────┬──────┬──┘   └──────────────────┘
                          │       │      │
             ┌────────────▼┐  ┌───▼────────────┐  ┌▼────────────────┐
             │ Scheme      │  │ Noise source   │  │ Step controller │
             │ step(...)   │  │ yields         │  │ fixed, adaptive │
             └──┬──────────┘  │ StepNoise      │  └─────────────────┘
                │             └───┬────────────┘
   ┌────────────▼──┐  ┌───────────▼──────┐
   │ Derivatives   │  │ Iterated         │
   │ user or SymPy │  │ integrals        │
   └───────────────┘  └──────────────────┘
   ───────────────────────────────────────────────────────────────
   Backend: Array API namespace and RNG (NumPy first, PyTorch later)
```

| Abstraction | Responsibility | Key interface |
| --- | --- | --- |
| `SDEProblem` / `ItoSDE` | Defines the equation; subclassed by the user | `dim`, `noise_dim`, `structure`, `x0`, `drift(t, x)`, `diffusion(t, x)`, optional `exact_solution(t, W)` |
| `SecondOrderSDE` | Second-order SODE in position/momentum form | `force(t, q, p)`, `diffusion(t, q, p)`, reduction to a first-order system |
| Derivative provider | Supplies derivatives needed by higher-order schemes | user-supplied methods or `SymbolicDerivatives` (SymPy) |
| `NoiseProcess` | Law of the driving noise | `WienerProcess(dim)`; later fBm, Lévy noise |
| `NoiseSource` | Delivers the random quantities of consecutive steps | `steps(grid) -> Iterator[StepNoise]` |
| `BrownianPath` | A materialized realization; one kind of `NoiseSource` | `refine`, `insert`, `coarsen`, `steps` |
| `StepNoise` | Everything random a scheme needs for one step | `h`, `dW`, on request `I` (double integrals), `Z` |
| `Scheme` | One update rule; declares orders and requirements | `step(problem, t, x, noise, state)`, `requires`, `init_state` |
| `StepController` | Chooses the next step size | `FixedStep`, `AdaptiveGrowthControl` |
| `Solver` / `Solution` | Runs the time loop over a batch of paths and stores the result | `solve(n_paths, seed)`, `Solution.t`, `Solution.x` |

### Noise: law, realization and step quantities

The noise model separates three things that a general simulation library may
blur, because strong convergence, adaptive step sizes and MLMC depend on
keeping them apart:

1. **The law of the noise** (`NoiseProcess`), for example an
   $m$-dimensional Wiener process. It has no state and no random numbers.
2. **A realization** of the noise on a probability space: a `BrownianPath`
   (materialized values on a grid) or a streaming source that produces the
   same realization block by block. A realization is fixed once created; see
   {doc}`../theory/brownian_motion`.
3. **The step quantities** (`StepNoise`): all random variables a scheme needs
   on one step $[t_n, t_{n+1}]$, computed from the realization. Euler–Maruyama
   needs only $h_n$ and $\Delta W_n$; Milstein for non-commutative noise also
   needs the double Itô integrals

   $$
   I_{ij}(s, t) = \int_s^t \int_s^r \mathrm{d}W^i(u) \, \mathrm{d}W^j(r) ,
   $$

   and Wagner–Platen needs $Z^j(s, t) = I_{(j,0)}(s, t) = \int_s^t \big(W^j(r) - W^j(s)\big)\,\mathrm{d}r$.

A scheme declares in `requires` which quantities it needs, and the source
computes only those. `Scheme.step` receives a `StepNoise` instead of a bare
`dW`, so higher-order schemes need no change of the interface.

**Coarsening rule.** Coupled step sizes must use the *same* realization, also
for the higher quantities. For $s \le t \le u$ the quantities on $[s, u]$
follow from those on $[s, t]$ and $[t, u]$ (Chen's relation):

$$
\begin{aligned}
\Delta W_{s,u} &= \Delta W_{s,t} + \Delta W_{t,u}, \\
I_{ij}(s, u) &= I_{ij}(s, t) + I_{ij}(t, u) + \Delta W^i_{s,t}\, \Delta W^j_{t,u}, \\
Z^j(s, u) &= Z^j(s, t) + Z^j(t, u) + (u - t)\, \Delta W^j_{s,t}.
\end{aligned}
$$

`StepNoise.coarsen` implements these identities, and the property-based tests
check them. Coarse quantities are never sampled independently of the fine
ones. When the fine double integrals are themselves approximations (truncated
Lévy-area expansions), the coarse values are computed from the fine
approximations by the same rule, so both levels stay consistent.

**Streaming instead of storing.** A materialized path of 100 000 paths, 16 384
steps and $m = 8$ needs about 98 GiB in double precision. The solver therefore
only requires a `NoiseSource` that yields `StepNoise` objects step by step.
`BrownianPath` is one implementation for teaching, tests and moderate studies;
a streaming source that draws increments in time blocks is the other. It must
reproduce `WienerProcess.sample_path` exactly for the same seed, independent
of the block size: both draw the normal samples in time order from one
generator, and a regression test fixes this. Splitting the paths across
parallel workers uses spawned seeds and is a different, documented realization.

### Coupling strategies

| Use | Required coupling | Provided by |
| --- | --- | --- |
| Strong convergence study | One realization for all step sizes | finest path, `coarsen` |
| Comparison of schemes | One realization for all schemes | the same source passed to each run |
| MLMC | Fine and coarse path coupled within a level; independent samples across levels and batches | per-level seeds spawned from one root seed; `coarsen` within a level |
| Antithetic MLMC | Fine path and its antithetic partner (swapped halves of the fine increments) on one level | `StepNoise` swap operation (phase 2) |
| Adaptive step sizes | Conditionally consistent, non-anticipating refinement of a single realization | Brownian bridge via `insert`; per-path grids in M3 |

### Noise structure

The structure of the diffusion selects fast code paths and decides which
schemes apply. The properties overlap, so they are combinable flags, not an
exclusive enum:

| Flag | Meaning |
| --- | --- |
| `scalar` | $m = 1$ |
| `additive` | $g(t, x)$ does not depend on $x$ |
| `diagonal` | $d = m$, $g$ is diagonal and $g_{ii}$ depends on $x_i$ only |
| `commutative` | $L^i g^j = L^j g^i$ for all $i, j$, where $L^j = \sum_k g_{kj}\,\partial_{x_k}$ |

Scalar, additive and diagonal noise are commutative; the library adds the
implied flags itself. Every flag is either **declared** by the user or
**verified** by the library, and a `NoiseStructure` object records which: a
symbolic problem can verify commutativity exactly with SymPy, a numerical
problem can be checked at sample points (a necessary condition only). Schemes
use fast paths for declared flags but report in the solution metadata whether
the flag was verified. Without flags the general code path is used, which is
always correct.

### Further conventions

- **Shapes.** `drift` returns `(..., d)`, `diffusion` returns `(..., d, m)`;
  the leading axes are the batch of paths. Stored trajectories have shape
  `(n_times, n_paths, d)`. A `save` option keeps only the final value or
  selected times for memory-heavy Monte Carlo runs.
- **Backend, dtype and device.** The solver takes namespace, dtype and device
  from the initial value `x0` if it is an array, otherwise NumPy and float64;
  explicit `backend`, `dtype` and `device` arguments override this. The noise
  source is created on the same backend and device. Problem methods must use
  `xp = array_namespace(x)` and create no arrays of their own on another
  device.
- **Scheme state.** A `Scheme` instance holds parameters only and can be
  reused across runs and threads. State that a run needs, such as the history
  of a multistep scheme, lives in an object returned by `scheme.init_state()`
  and owned by the solver run; one-step schemes return `None`.
- **Capability checks.** Before the time loop, the solver checks that the
  scheme's requirements (derivatives, step quantities, noise structure,
  calculus) are met and fails early with a readable message.
- **Calculus.** Problems carry a `calculus` attribute, `ITO` for now; a later
  `STRATONOVICH` value plugs in without touching existing schemes.
- **Random numbers.** Drawn by NumPy's PCG64 generator from immutable seed
  sequences and converted to the target namespace, so a seed gives identical
  numbers on every backend.
- **Iterated integrals.** Exact formulas for scalar and diagonal noise;
  products of increments for commutative noise; for general noise one
  well-tested algorithm in the teaching release (see the roadmap) and further
  algorithms as experimental options.

## API sketch

The names below are a sketch for M1. The example becomes a runnable
integration test in M1; until then it documents intent, not working code.

```python
import numsde as ns


class GeometricBrownianMotion(ns.ItoSDE):
    """dX = mu X dt + sigma X dW, with exact solution."""

    structure = ns.NoiseStructure(scalar=True)

    def __init__(self, mu=0.05, sigma=0.2, x0=1.0):
        super().__init__(dim=1, noise_dim=1, x0=x0)
        self.mu, self.sigma = mu, sigma

    def drift(self, t, x):
        return self.mu * x

    def diffusion(self, t, x):
        return (self.sigma * x)[..., None]  # shape (..., d, m)

    def exact_solution(self, t, W):
        xp = ns.array_namespace(W)
        return self.x0 * xp.exp((self.mu - 0.5 * self.sigma**2) * t + self.sigma * W)


problem = GeometricBrownianMotion()
path = ns.WienerProcess(1).sample_path(
    ns.uniform_grid(1.0, 2**10), n_paths=5_000, seed=1
)

solution = ns.Solver(problem, ns.schemes.EulerMaruyama()).solve(path.coarsen(4))

study = ns.analysis.StrongConvergenceStudy(
    problem,
    schemes=[ns.schemes.EulerMaruyama()],
    path=path,  # finest realization; coarsened per level
    factors=[2**k for k in range(1, 7)],
    reference="exact",  # uses problem.exact_solution on the same path
)
result = study.run()
result.orders  # fitted strong order per scheme, with bootstrap confidence interval
```

## Method catalog

The teaching release covers the schemes of milestones M1–M3. The orders are the
classical results under the assumptions of the cited reference; they are not
unconditional. Each theory page states the precise assumptions (smoothness and
growth of the coefficients, noise structure, error measure) with a reference to
the theorem. Entries marked † still have to be checked against the reference.

| Scheme | Family | Strong / weak order | Needs | Milestone |
| --- | --- | --- | --- | --- |
| Euler–Maruyama | Explicit one-step | 0.5 (1.0 for additive noise) / 1.0 | — | M1 |
| Milstein, scalar / diagonal / commutative noise | Explicit one-step | 1.0 / 1.0 | diffusion derivatives | M2 |
| Milstein, general noise | Explicit one-step | 1.0 / 1.0 | derivatives, iterated integrals | M2 (one Lévy-area algorithm); further algorithms experimental |
| Wagner–Platen (order 1.5 Taylor scheme), scalar and additive noise | Explicit one-step, higher order | 1.5 / — | first and second derivatives; correlated pair (ΔW, ΔZ) | M2 |
| Derivative-free Milstein (Runge–Kutta type) | Explicit one-step | 1.0 / 1.0 | iterated integrals for general noise | M2 |
| Tamed Euler–Maruyama (Hutzenthaler, Jentzen, Kloeden) | Superlinear coefficients | 0.5 / — | — | M3 |
| Tamed Milstein (Wang, Gan) | Superlinear coefficients | 1.0 / — | derivatives | M3 |
| Truncated Euler–Maruyama (Mao) | Superlinear coefficients | up to 0.5 / — | growth function | M3 |
| Projected Euler–Maruyama / Milstein (Beyn, Isaak, Kruse) | Superlinear coefficients | 0.5 (EM), 1.0 (Milstein) / — | projection radius | M3 |
| Adaptive growth-controlling Euler–Maruyama (Kelly, Lord) | Adaptive stepping | 0.5 / — | Brownian bridge refinement | M3 |
| Stochastic Runge–Kutta SRI / SRA (Rößler) | Explicit one-step, higher order | 1.5 (scalar / additive noise) / 2.0 † | — | Phase 2 (M4) |
| Stochastic θ-method, backward Euler–Maruyama | Implicit one-step | 0.5 / 1.0 | nonlinear solver | Phase 2 |
| BDF2-Maruyama (Andersson, Kruse) | Implicit two-step | 0.5 in mean square under a global monotonicity condition / — | nonlinear solver, starting step (backward Euler–Maruyama) | Phase 2 (M4) |
| Further stochastic linear multistep schemes (Buckwar, Winkler) | Explicit and implicit multistep | scheme-dependent | starting procedure; nonlinear solver if implicit | Phase 2 (M4) |
| Stochastic symplectic Euler and Störmer–Verlet type (Milstein, Repin, Tretyakov) | Geometric, 2nd-order SODEs | scheme-dependent | separable Hamiltonian structure | Phase 2 (M4) |
| Splitting schemes for Langevin dynamics (e.g. BAOAB) | Geometric, 2nd-order SODEs | — / 2.0 for the invariant measure | Langevin structure | Phase 2 (M4) |
| Derivative-free higher-order schemes via multistep differences | Explicit multistep | to be specified | history buffer | Phase 2 |
| MLMC estimator, incl. antithetic variant (Giles, Szpruch) | Estimator | — | coupled coarse / fine paths | Phase 2 |

## Convergence studies, benchmark suite and MLMC

All three build on one mechanism: the same Brownian path, sampled once on the
finest grid and coarsened, drives every scheme and every step size being
compared.

**Convergence studies** (`numsde.analysis`). The default strong error is the
error uniformly on the grid, matching the error notion of the book project; the
final-time error is available as an option:

$$
\begin{aligned}
e_h^{\mathrm{strong}} &= \Big(\mathbb{E}\Big[\max_{n \in \{0,\dots,N\}} \big|X_h^n - X(t_n)\big|^p\Big]\Big)^{1/p} && \text{(default)}\\
e_h^{\mathrm{strong},T} &= \Big(\mathbb{E}\,\big|X_h^N - X(T)\big|^p\Big)^{1/p} && \text{(final time)}\\
e_h^{\mathrm{weak}} &= \big|\,\mathbb{E}\,\varphi(X_h^N) - \mathbb{E}\,\varphi(X(T))\,\big|
\end{aligned}
$$

- The maximum over the grid is accumulated as a running maximum during the
  time loop, so no trajectories need to be stored. The reference solution is
  needed at every grid point: the exact solution on the same Brownian path, or
  the fine-grid reference restricted to the coarse grid.
- Results are plain data objects with an optional `plot()`.

**Statistics of a convergence study.** Monte Carlo estimation and the fit of
the order are kept apart:

1. *Per-path data.* The study stores, for every path $i$ and step size $h_k$,
   the pathwise quantity $Y_{ik} = \max_n |X_{h_k}^n - X(t_n)|^p$ (strong) or
   $\varphi(X_{h_k}^N)$ (weak). This matrix is small compared with the paths.
2. *Error estimates.* Per step size the mean, its standard error and a
   confidence interval; the strong error $(\overline{Y}_k)^{1/p}$ gets its
   interval by transforming the interval of the mean.
3. *Order fit.* Least squares on $\log e_{h_k}$ against $\log h_k$. All step
   sizes use the same paths, so the estimates are correlated and the naive
   regression interval is wrong. The confidence interval of the order comes
   from a pathwise bootstrap: resample paths (the same indices for all step
   sizes), recompute all errors, refit.
4. *Weak errors.* First the signed difference
   $D_k = \mathbb{E}\,\varphi(X_{h_k}^N) - \mathbb{E}\,\varphi(X(T))$ is
   estimated with its confidence interval; only then is the absolute value
   reported. Taking absolute values of individual estimates first would bias
   small errors upwards. When the interval of $D_k$ contains zero, the study
   flags that the Monte Carlo error dominates.

**Benchmark suite** (`numsde.benchmarks`). A registry of test problems with
metadata (dimension, noise type, exact solution or reference, structural
properties): geometric Brownian motion, Ornstein–Uhlenbeck, linear systems
with commutative and non-commutative noise, a stochastic Ginzburg–Landau
equation, a double-well problem, the Kubo oscillator and the stochastic
pendulum. A runner compares error and cost (wall time, number of coefficient
evaluations) of a new scheme against reference schemes.

Every benchmark problem has its own theory page under `docs/theory/problems/`,
analogous to the scheme pages under `docs/theory/schemes/`. It states

- the equation with its parameters, their default values, and the dimensions
  $d$ and $m$;
- the noise structure (scalar, additive, diagonal, commutative or general);
- the exact solution, or how the reference solution is computed and how
  accurate it is;
- the structural properties: which standard assumptions hold or fail (global
  Lipschitz, superlinear growth, one-sided Lipschitz), moment bounds,
  invariants and conserved or dissipated quantities;
- what the problem is meant to test, with the expected strong and weak orders
  of the relevant schemes and known failure modes (e.g. divergence of
  Euler–Maruyama for superlinear coefficients);
- references, and a link to the registry entry.

**Multilevel Monte Carlo** (`numsde.mlmc`). Level estimators from coupled
coarse and fine paths, level selection and sample allocation following Giles,
and per-level mean and variance decay, which show weak and strong order side by
side.

## Quality assurance

| Layer | Tool | What it checks | When it runs |
| --- | --- | --- | --- |
| Unit tests | pytest | single functions and scheme steps against hand-computed values | every push and PR |
| Property-based tests | Hypothesis | invariants for all shapes and parameters | every push and PR |
| Doctests | pytest `--doctest-modules` | every docstring example runs | every push and PR |
| Statistical tests | pytest, marker `slow` | empirical orders match theory within a tolerance, fixed seeds | nightly and before release |
| Regression tests | pytest + stored reference data | results reproduce numbers from the legacy scripts | every push and PR |
| Notebooks | myst-nb execution in the docs build | tutorials run end to end | docs build on every PR |

- Ruff for linting and formatting; mypy in strict mode on `src/`; codespell
  for US spelling; pre-commit runs all three before each commit.
- Line and branch coverage with a target of 100 %.
- GitHub-hosted runners have no GPU: the PyTorch path is tested on CPU in CI.
  GPU tests carry the `gpu` marker, are skipped in CI and run on the
  maintainer's local machine (`nox -s gpu`). A passing local GPU run is part of
  every release gate from the first PyTorch release on.
- CI matrix: CPython 3.11–3.14 on Ubuntu, the newest version on macOS and
  Windows, PyPy 3.11 with the NumPy backend, a minimal-dependency job and an
  all-extras job. `nox` sessions mirror the CI jobs.

## Documentation

Sphinx with MyST-Markdown, hosted on Read the Docs with versioned
documentation per release. Anything that can drift from the code is generated
or executed:

1. API pages are generated from NumPy-style docstrings.
2. Docstring examples run as doctests.
3. Notebooks execute during the build; a broken tutorial breaks the build.
4. `sphinx-build -W` turns every warning into an error; a scheduled
   `linkcheck` job catches dead external links.
5. The definition of done for any scheme and any benchmark problem includes
   its theory page.

{doc}`../errata` is the single place where corrected errors are documented.
The changelog, release notes, issues and theory pages only link to an errata
entry by its ID.

## Repository, tooling and releases

| Topic | Decision |
| --- | --- |
| Repository | `github.com/mlu-numerics/numsde`; `main` protected, every change through a pull request with green CI |
| License | BSD-3-Clause |
| Python | ≥ 3.11 |
| Core dependencies | NumPy, `array-api-compat`; SciPy only once a module needs it |
| Optional extras | `[symbolic]` SymPy, `[plot]` matplotlib, `[torch]` PyTorch, `[all]` |
| Dev dependency groups | `test`, `lint`, `docs`, `dev` |
| Build | `hatchling` with `hatch-vcs` (versions from Git tags) |
| Environments | `uv`; `nox` for sessions; `pre-commit` hooks |
| Versioning | Semantic Versioning, `0.x` until the API is stable; `CHANGELOG.md` in Keep a Changelog format, fixes link to their errata entry |
| Publishing | PyPI via Trusted Publishing from a tagged release workflow |
| Documentation hosting | Read the Docs |
| Citation | `CITATION.cff`; Zenodo DOI per release once the core is stable |

**Legacy scripts.** The 63 scripts of the book project (45 Python, 18 MATLAB)
were inventoried in October 2026; 32 are SDE-related, the rest (random number
generation, Monte Carlo basics, KL and fBm) stay with the book. They are ported
in order of milestone. Reference numbers from seeded NumPy scripts are stored
once as `.npz` files with seed, parameters and script version; MATLAB results
serve only as statistical plausibility checks. Ported scripts land in
`examples/`.

**Book project.** No book text in the repository. Library and book share only
the bibliography and reproducible figure scripts; figures are generated from
tagged library versions.

## Roadmap

Five milestones lead to the teaching release v0.4 on 1 September 2027; M4 and
the remaining items follow in phase 2 after the lecture.

| Milestone | Period | Deliverables | Release |
| --- | --- | --- | --- |
| M0 Foundations (done) | Oct 2026 | repository, packaging, CI, pre-commit, docs skeleton, backend layer, `WienerProcess` and `BrownianPath`, legacy-script inventory | — |
| M1 Core and Euler–Maruyama | Nov 2026 – Jan 2027 | `ItoSDE`, `NoiseStructure`, `StepNoise` and `NoiseSource`, `Solver`, `Solution`, fixed grids, Euler–Maruyama; first an end-to-end slice with GBM and a linear system with multidimensional noise, then the strong convergence study with bootstrap intervals, Brownian motion on the circle as third test problem, first tutorial; ports of the GBM scripts; PyTorch CPU job in CI | v0.1 |
| M2 Milstein and iterated integrals | Feb – Mar 2027 | Core: Milstein for scalar, diagonal and commutative noise; iterated-integrals module with Chen coarsening and one well-tested Lévy-area algorithm for general noise; SymPy derivatives; derivative-free Milstein; Wagner–Platen for scalar and additive noise. Experimental, not required for the teaching release: Wiktorsson and Mrongowius–Rößler algorithms | v0.2 |
| M3 Superlinear coefficients | Apr – Jun 2027 | tamed, truncated and projected schemes, per-path grids and Kelly–Lord adaptive stepping, evaluation of Brownian trees; test problems stochastic Ginzburg–Landau, 2-D stochastic oscillator, stochastic logistic equation, volatility model | v0.3 |
| M5 Teaching release | Jul – Aug 2027 | API review, theory pages, tutorials, remaining legacy ports, PyPI release, Zenodo DOI | v0.4 |
| M4 Higher order and geometric | Phase 2 | Rößler SRK, stochastic θ-method, BDF2-Maruyama (Andersson–Kruse) and further linear multistep schemes (Buckwar–Winkler), `SecondOrderSDE`, symplectic and splitting schemes, weak convergence studies, benchmark runner | v0.5 |
| Phase 2, further items | after the lecture | MLMC incl. antithetic variant, GPU performance, derivative-free multistep variants, finite-difference and autodiff derivatives, Stratonovich | towards v1.0 |
| Phase 3 | later | fractional Brownian motion, Lévy noise, rough paths, implicit schemes | v1.x |

**Gate for every milestone**

- All schemes of the milestone pass unit, property and statistical tests.
- Coverage at target, mypy and Ruff clean, docs build with `-W`.
- Each new scheme has a theory page, a docstring example and a benchmark entry.
- Each new benchmark problem has a theory page and a registry entry.
- At least one legacy script is ported onto the new features.

## Risks and open questions

| Risk | Mitigation |
| --- | --- |
| Scope exceeds the time of a single developer | M4 moved to phase 2; M3 has three months; further slips cut features, never tests or theory pages |
| Iterated integrals for general noise are costly and hard to verify | M2 split into a tested core and experimental algorithms; Chen coarsening tested by properties; antithetic MLMC as alternative |
| Path semantics not suited for adaptive per-path grids | M3 evaluates Brownian trees before the adaptive solver interface is fixed |
| Array API gaps between NumPy and PyTorch | CPU PyTorch job in CI from M1; backend-specific code only in `_backend/` |
| Flaky statistical tests | fixed seeds, tolerances derived from the Monte Carlo error, slow suite separate |
| API designed in the abstract | port legacy scripts early, as an API usability test |

**Open questions**

- Positioning against existing packages (`sdeint`, `torchsde`, Diffrax,
  Julia's StochasticDiffEq).
- Exact scope of the derivative-free higher-order schemes based on multistep
  differences.
- Which Lévy-area algorithm forms the tested core of M2.
- Whether the streaming source adds counter-based sampling, so that results
  are also independent of how paths are split across workers.

## Decision log

| Date | Decision |
| --- | --- |
| 2026-10-23 | Package and project name `numsde`; repository under `mlu-numerics` |
| 2026-10-23 | BSD-3-Clause license |
| 2026-10-23 | Sphinx + MyST-Markdown, NumPy-style docstrings |
| 2026-10-23 | Object-oriented API; users define problems as classes |
| 2026-10-23 | NumPy first, PyTorch through an Array API backend abstraction |
| 2026-10-23 | Itô calculus first; architecture open for Stratonovich, fBm, Lévy noise, rough paths |
| 2026-10-23 | Python ≥ 3.11; pytest, Hypothesis, Ruff, mypy, GitHub Actions |
| 2026-10-24 | M4 moves to phase 2; teaching release becomes v0.4 |
| 2026-10-24 | Geometric integration references: {cite}`HairerLubichWanner2006` (deterministic), Milstein and Tretyakov (stochastic) |
| 2026-10-24 | Default strong error is the error uniformly on the grid |
| 2026-10-24 | `docs/errata.md` is the single source for documented corrections |
| 2026-10-24 | GPU tests run locally on the maintainer's machine, not in CI |
| 2026-10-24 | Documentation hosted on Read the Docs |
| 2026-10-24 | This page becomes the authoritative architecture document; the planning draft it came from is frozen |
| 2026-10-24 | SciPy is not a dependency until a module needs it; keeps PyPy installable from wheels |
| 2026-10-24 | Random numbers drawn with NumPy PCG64 and converted per backend, so seeds reproduce across backends |
| 2026-10-25 | `BrownianPath` stores all paths of a batch on one common grid and is immutable (refine, insert and coarsen return new paths); per-path grids for adaptive step sizes are added in M3 |
| 2026-10-25 | US English throughout code and documentation, matching the book project; enforced with codespell |
| 2026-10-26 | A `BrownianPath` is a fixed realization: values owned and immutable, an immutable seed instead of a generator, `insert`/`refine` deterministic in seed and requested times (audit findings A1, A2); Brownian trees are evaluated for M3 |
| 2026-10-26 | Wrong argument types raise `TypeError`, wrong values `ValueError`; booleans are never accepted as integers |
| 2026-10-26 | Noise model in three layers: law (`NoiseProcess`), realization (`BrownianPath` or a streaming `NoiseSource`), step quantities (`StepNoise`); schemes receive `StepNoise`, and coarse step quantities follow from fine ones by Chen's relation (audit A4, A6) |
| 2026-10-26 | The solver depends on a `NoiseSource`, not on a materialized path; a streaming source must reproduce `sample_path` for the same seed independent of the block size |
| 2026-10-26 | Coupling strategies per use case, MLMC with independent levels (audit A5) |
| 2026-10-26 | `NoiseStructure` flags (scalar, additive, diagonal, commutative), declared or verified, replace the `NoiseType` enum (audit A8) |
| 2026-10-26 | Backend, dtype and device follow the initial value; scheme instances are stateless, run state comes from `init_state` (audit A11, A12) |
| 2026-10-26 | Convergence statistics: per-path data, pathwise bootstrap for order intervals, signed weak bias before absolute values (audit A7) |
| 2026-10-26 | M2 split into a core for the teaching release and experimental Lévy-area algorithms (audit A14); stochastic θ-method and BDF2-Maruyama added to phase 2; test problems from the script inventory added to M1 and M3 |
| 2026-10-26 | M0 completed |
| 2026-10-27 | BDF2-Maruyama is listed as an implicit two-step scheme with reference {cite}`AnderssonKruse2017`; implicit multistep schemes come with the implicit schemes in phase 2 |
| 2026-10-27 | Notation: processes as $X(t)$, $W(t)$; subscripts only for discretizations and increments, as in the book project |
| 2026-10-27 | Every benchmark problem gets its own theory page under `docs/theory/problems/`, analogous to the scheme pages under `docs/theory/schemes/`; part of the definition of done |

## References

Literature for the planned schemes, by topic. Entries move into
`docs/references.bib` once their details have been checked.

- *Foundations and iterated integrals:* Kloeden and Platen (1992); Kloeden,
  Platen and Wright (1992); Wiktorsson (2001); Mrongowius and Rößler (2022);
  Rößler (2010).
- *Superlinear coefficients and adaptivity:* Hutzenthaler, Jentzen and Kloeden
  (2012); Wang and Gan (2013); Mao (2015); Beyn, Isaak and Kruse (2016, 2017);
  Kelly and Lord (2018).
- *Multistep and geometric schemes:* Andersson and Kruse (2017); Buckwar and Winkler (2006); Milstein,
  Repin and Tretyakov (2002); Milstein and Tretyakov (2021); Leimkuhler and
  Matthews (2013); Hairer, Lubich and Wanner (2006).
- *Multilevel Monte Carlo:* Giles (2008); Giles and Szpruch (2014).
