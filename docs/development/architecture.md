# Architecture and roadmap

This page is the authoritative description of NumSDE's scope, architecture and
roadmap. Changes go through pull requests like any code change; decisions are
recorded in the [decision log](#decision-log) at the end.

## Vision and scope

NumSDE is a BSD-3-licensed Python library of numerical methods for Itô SDEs,
built for university teaching and research. A teaching release must be ready by
1 September 2027, ahead of the next lecture on the topic.

$$
\mathrm{d}X_t = f(t, X_t)\,\mathrm{d}t + g(t, X_t)\,\mathrm{d}W_t,
\qquad X_t \in \mathbb{R}^d,\; W_t \in \mathbb{R}^m,\; g(t,x) \in \mathbb{R}^{d \times m}
$$

**Goals**

- Readable, verified reference implementations of explicit one-step and
  multistep schemes for multidimensional SDEs with general noise.
- Tools for empirical strong and weak convergence studies, plus a benchmark
  suite to test new schemes against established ones from the literature.
- Didactic documentation: a theory page for every scheme and executable
  tutorials.
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
   convergence tests, a docstring with a runnable example, and a theory page.

## Package layout

```text
numsde/
├── src/numsde/
│   ├── _backend/          # array namespace resolution, RNG wrapper
│   ├── problems/          # SDEProblem, ItoSDE, NoiseType, SecondOrderSDE,
│   │                      # SymbolicSDE, derivative providers
│   ├── noise/             # WienerProcess, BrownianPath, refinement, coarsening
│   ├── integrals/         # iterated stochastic integrals and Lévy areas
│   ├── schemes/
│   │   ├── base.py        # Scheme, OneStepScheme, MultiStepScheme
│   │   ├── explicit/      # Euler–Maruyama, Milstein, Wagner–Platen, SRK
│   │   ├── nonlinear/     # tamed, truncated, projected schemes
│   │   ├── multistep/     # Buckwar–Winkler type schemes
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
             │ Scheme      │  │ Noise process  │  │ Step controller │
             │ step(...)   │  │ refine/coarsen │  │ fixed, adaptive │
             └──┬───────┬──┘  └───────┬────────┘  └─────────────────┘
                │       │             │
   ┌────────────▼──┐  ┌─▼─────────────▼──┐
   │ Derivatives   │  │ Iterated         │
   │ user or SymPy │  │ integrals        │
   └───────────────┘  └──────────────────┘
   ───────────────────────────────────────────────────────────────
   Backend: Array API namespace and RNG (NumPy first, PyTorch later)
```

| Abstraction | Responsibility | Key interface |
| --- | --- | --- |
| `SDEProblem` / `ItoSDE` | Defines the equation; subclassed by the user | `dim`, `noise_dim`, `noise_type`, `x0`, `drift(t, x)`, `diffusion(t, x)` |
| `SecondOrderSDE` | Second-order SODE in position/momentum form | `force(t, q, p)`, `diffusion(t, q, p)`, reduction to a first-order system |
| Derivative provider | Supplies derivatives needed by higher-order schemes | user-supplied methods or `SymbolicDerivatives` (SymPy) |
| `NoiseProcess` / `BrownianPath` | Samples increments; refines by Brownian bridge, coarsens by summation | `increments(grid, n_paths)`, `refine()`, `coarsen(factor)` |
| Iterated integrals | Samples double Itô integrals and Lévy areas consistent with given increments | `sample(dW, h, method=...)` |
| `Scheme` | One update rule; declares orders and requirements | `step(t, x, h, dW, ...)`, `strong_order`, `weak_order`, `requires` |
| `StepController` | Chooses the next step size | `FixedStep`, `AdaptiveGrowthControl` |
| `Solver` / `Solution` | Runs the time loop over a batch of paths and stores the result | `solve(n_paths, seed)`, `Solution.t`, `Solution.x` |

**Conventions**

- **Shapes.** `drift` returns `(..., d)`, `diffusion` returns `(..., d, m)`;
  the leading axes are the batch of paths. Stored trajectories have shape
  `(n_times, n_paths, d)`. A `save` option keeps only the final value or
  selected times for memory-heavy Monte Carlo runs.
- **Noise types.** `NoiseType` is one of `SCALAR`, `ADDITIVE`, `DIAGONAL`,
  `COMMUTATIVE`, `GENERAL`. Schemes dispatch on it to specialized fast paths;
  `GENERAL` is always correct.
- **Capability checks.** Before the time loop, the solver checks that the
  scheme's requirements (derivatives, iterated integrals, noise type, calculus)
  are met and fails early with a readable message.
- **Multistep schemes.** `MultiStepScheme` owns a history buffer and a starting
  procedure, so the solver treats both scheme kinds alike.
- **Calculus.** Problems carry a `calculus` attribute, `ITO` for now; a later
  `STRATONOVICH` value plugs in without touching existing schemes.
- **Backend.** The array namespace is resolved from the input arrays via
  `array-api-compat`. Random numbers are drawn by NumPy's PCG64 generator with
  `SeedSequence` spawning and converted to the target namespace, so a seed
  gives identical numbers on every backend.
- **Iterated integrals.** Exact formulas for scalar and diagonal noise;
  products of increments for commutative noise; for general noise the Fourier
  method (Kloeden–Platen–Wright), Wiktorsson's method and the
  Mrongowius–Rößler algorithm, selectable per run.

## API sketch

The names below are a sketch to be validated by porting the existing lecture
scripts.

```python
import numsde as ns


class GeometricBrownianMotion(ns.ItoSDE):
    """dX = mu X dt + sigma X dW."""

    noise_type = ns.NoiseType.DIAGONAL

    def __init__(self, mu=0.05, sigma=0.2, x0=1.0):
        super().__init__(dim=1, noise_dim=1, x0=x0)
        self.mu, self.sigma = mu, sigma

    def drift(self, t, x):
        return self.mu * x

    def diffusion(self, t, x):
        return (self.sigma * x)[..., None]  # shape (..., d, m)


problem = GeometricBrownianMotion()
solver = ns.Solver(problem, scheme=ns.schemes.Milstein(), dt=2**-8, t_end=1.0)
solution = solver.solve(n_paths=10_000, seed=42)

study = ns.analysis.StrongConvergenceStudy(
    problem,
    schemes=[ns.schemes.EulerMaruyama(), ns.schemes.Milstein()],
    reference=problem.exact_solution,
    dts=[2**-k for k in range(4, 11)],
    n_paths=5_000,
    seed=1,
)
result = study.run()
result.orders  # fitted strong order per scheme, with confidence intervals
```

## Method catalog

The teaching release covers the schemes of milestones M1–M3; orders are the
classical results under the standard assumptions of each reference.

| Scheme | Family | Strong / weak order | Needs | Milestone |
| --- | --- | --- | --- | --- |
| Euler–Maruyama | Explicit one-step | 0.5 (1.0 for additive noise) / 1.0 | — | M1 |
| Milstein, scalar / diagonal / commutative noise | Explicit one-step | 1.0 / 1.0 | diffusion derivatives | M2 |
| Milstein, general noise | Explicit one-step | 1.0 / 1.0 | derivatives, iterated integrals | M2 |
| Wagner–Platen (order 1.5 Taylor scheme), scalar and additive noise | Explicit one-step, higher order | 1.5 / 2.0 | first and second derivatives; correlated pair (ΔW, ΔZ) | M2 |
| Derivative-free Milstein (Runge–Kutta type) | Explicit one-step | 1.0 / 1.0 | iterated integrals for general noise | M2 |
| Tamed Euler–Maruyama (Hutzenthaler, Jentzen, Kloeden) | Superlinear coefficients | 0.5 / — | — | M3 |
| Tamed Milstein (Wang, Gan) | Superlinear coefficients | 1.0 / — | derivatives | M3 |
| Truncated Euler–Maruyama (Mao) | Superlinear coefficients | up to 0.5 / — | growth function | M3 |
| Projected Euler–Maruyama / Milstein (Beyn, Isaak, Kruse) | Superlinear coefficients | 0.5 (EM), 1.0 (Milstein) / — | projection radius | M3 |
| Adaptive growth-controlling Euler–Maruyama (Kelly, Lord) | Adaptive stepping | 0.5 / — | Brownian bridge refinement | M3 |
| Stochastic Runge–Kutta SRI / SRA (Rößler) | Explicit one-step, higher order | 1.5 (scalar / additive noise) / 2.0 | — | Phase 2 (M4) |
| Stochastic linear multistep schemes (Buckwar, Winkler) | Explicit multistep | scheme-dependent | starting procedure | Phase 2 (M4) |
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
- Orders are estimated by least squares on the log-log data and reported with
  a confidence interval from the Monte Carlo standard error.
- Weak studies report the Monte Carlo error next to the discretization error.
- Results are plain data objects with an optional `plot()`.

**Benchmark suite** (`numsde.benchmarks`). A registry of test problems with
metadata (dimension, noise type, exact solution or reference, structural
properties): geometric Brownian motion, Ornstein–Uhlenbeck, linear systems
with commutative and non-commutative noise, a stochastic Ginzburg–Landau
equation, a double-well problem, the Kubo oscillator and the stochastic
pendulum. A runner compares error and cost (wall time, number of coefficient
evaluations) of a new scheme against reference schemes.

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
5. The definition of done for any scheme includes its theory page.

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

**Legacy scripts.** The existing Python 2.7, Python 3 and Matlab scripts are
inventoried and ported in order of lecture relevance. Matlab results are stored
as reference data for regression tests; ported scripts land in `examples/`.

**Book project.** No book text in the repository. Library and book share only
the bibliography and reproducible figure scripts; figures are generated from
tagged library versions.

## Roadmap

Five milestones lead to the teaching release v0.4 on 1 September 2027; M4 and
the remaining items follow in phase 2 after the lecture.

| Milestone | Period | Deliverables | Release |
| --- | --- | --- | --- |
| M0 Foundations | Nov 2026 | repository, packaging, CI, pre-commit, docs skeleton, backend layer, `WienerProcess` and `BrownianPath`, legacy-script inventory | — |
| M1 Core and Euler–Maruyama | Dec 2026 – Jan 2027 | `ItoSDE`, `Solver`, `Solution`, fixed grids, Euler–Maruyama, first test problems, strong convergence study, first tutorial; PyTorch CPU job in CI | v0.1 |
| M2 Milstein and iterated integrals | Feb – Mar 2027 | Milstein for all noise types, iterated-integrals module, SymPy derivatives, derivative-free Milstein, Wagner–Platen for scalar and additive noise | v0.2 |
| M3 Superlinear coefficients | Apr – Jun 2027 | tamed, truncated and projected schemes, Brownian bridge refinement, Kelly–Lord adaptive stepping | v0.3 |
| M5 Teaching release | Jul – Aug 2027 | API review, theory pages, tutorials, remaining legacy ports, PyPI release, Zenodo DOI | v0.4 |
| M4 Higher order and geometric | Phase 2 | Rößler SRK, Buckwar–Winkler multistep schemes, `SecondOrderSDE`, symplectic and splitting schemes, weak convergence studies, benchmark runner | v0.5 |
| Phase 2, further items | after the lecture | MLMC incl. antithetic variant, GPU performance, derivative-free multistep variants, finite-difference and autodiff derivatives, Stratonovich | towards v1.0 |
| Phase 3 | later | fractional Brownian motion, Lévy noise, rough paths, implicit schemes | v1.x |

**Gate for every milestone**

- All schemes of the milestone pass unit, property and statistical tests.
- Coverage at target, mypy and Ruff clean, docs build with `-W`.
- Each new scheme has a theory page, a docstring example and a benchmark entry.
- At least one legacy script is ported onto the new features.

## Risks and open questions

| Risk | Mitigation |
| --- | --- |
| Scope exceeds the time of a single developer | M4 moved to phase 2; M3 has three months; further slips cut features, never tests or theory pages |
| Iterated integrals for general noise are costly | separate module, several algorithms selectable, antithetic MLMC as alternative |
| Array API gaps between NumPy and PyTorch | CPU PyTorch job in CI from M1; backend-specific code only in `_backend/` |
| Flaky statistical tests | fixed seeds, tolerances derived from the Monte Carlo error, slow suite separate |
| API designed in the abstract | port legacy scripts early, as an API usability test |

**Open questions**

- Positioning against existing packages (`sdeint`, `torchsde`, Diffrax,
  Julia's StochasticDiffEq).
- Exact scope of the derivative-free higher-order schemes based on multistep
  differences.

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

## References

Literature for the planned schemes, by topic. Entries move into
`docs/references.bib` once their details have been checked.

- *Foundations and iterated integrals:* Kloeden and Platen (1992); Kloeden,
  Platen and Wright (1992); Wiktorsson (2001); Mrongowius and Rößler (2022);
  Rößler (2010).
- *Superlinear coefficients and adaptivity:* Hutzenthaler, Jentzen and Kloeden
  (2012); Wang and Gan (2013); Mao (2015); Beyn, Isaak and Kruse (2016, 2017);
  Kelly and Lord (2018).
- *Multistep and geometric schemes:* Buckwar and Winkler (2006); Milstein,
  Repin and Tretyakov (2002); Milstein and Tretyakov (2021); Leimkuhler and
  Matthews (2013); Hairer, Lubich and Wanner (2006).
- *Multilevel Monte Carlo:* Giles (2008); Giles and Szpruch (2014).
