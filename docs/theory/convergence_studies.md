# Convergence studies

An empirical convergence study estimates the error of a scheme for several
step sizes and fits the order of convergence. This page describes how
{class}`numsde.analysis.StrongConvergenceStudy` does this and how reliable the
reported intervals are. The estimators are available individually in
`numsde.analysis.statistics` (see the {doc}`API reference <../api/index>`).

## One realization for all step sizes

All schemes and step sizes are driven by the same Brownian path: it is sampled
once on the finest grid, and the grid of step size $h_k$ is obtained by
coarsening with a factor $k$, which sums the fine increments exactly (see
{doc}`brownian_motion`). The errors of different step sizes are therefore
computed on the same paths. This removes the Monte Carlo noise *between* step
sizes, which would otherwise blur the slope, but it makes the estimates for
different step sizes correlated, which matters for the intervals below.

## Error and reference solution

For $p \ge 1$ the study estimates the strong error uniformly on the grid,

$$
e_h = \Big(\mathbb{E}\Big[\max_{n} \big|X_h^n - X(t_n)\big|^p\Big]\Big)^{1/p},
$$

or, with `error="final"`, the error $\big(\mathbb{E}\,|X_h^N - X(T)|^p\big)^{1/p}$
at the final time. The reference values $X(t_n)$ come from

- the exact solution of the problem on the same path (`reference="exact"`), or
- a scheme run on the path refined by the Brownian bridge with
  `reference_refinement` substeps per step, restricted to the coarse grids.
  Its grid must be strictly finer than every grid of the study, and its own
  error should be much smaller than the smallest error of the study.

The maximum over the grid is accumulated during the time loop
({meth}`numsde.Solver.iterate`), so no trajectories are stored. Only a scheme
reference is kept, at the points of the coarse grids of the study, which need
not be nested (factors 2 and 3, say).

## Error estimates

For path $i = 1, \dots, n$ and step size $h_k$ the study records

$$
Y_{ik} = \max_n \big|X_{h_k}^n - X(t_n)\big|^p .
$$

The error is estimated as $\hat e_k = \bar Y_k^{1/p}$ with the sample mean
$\bar Y_k$. Its interval transforms the normal interval
$\bar Y_k \pm z_{1-\alpha/2}\, s_k/\sqrt{n}$ of the mean with
$y \mapsto y^{1/p}$, which is monotone; a negative lower limit is replaced by
$0$. The experimental order of convergence between neighboring step sizes is
$\log(\hat e_{k+1}/\hat e_k) / \log(h_{k+1}/h_k)$.

## Order fit

The fitted order $\hat q$ is the slope of the least-squares line through the
points $(\log h_k, \log \hat e_k)$. By default all step sizes enter the fit;
coarse step sizes outside the asymptotic range can be excluded with
`fit_factors`, also afterwards with
{meth}`~numsde.analysis.StrongConvergenceResult.refit`, which does not
simulate again.

The regression interval of ordinary least squares does not apply: the points
are estimates from the same paths, hence correlated, and the residuals of a
convergence plot are mostly systematic (higher-order terms), not random. The
interval of $\hat q$ therefore comes from a pathwise bootstrap.

## Bootstrap intervals

A bootstrap replicate draws $n$ path indices with replacement and uses them for
**all** step sizes, which preserves the correlation between step sizes. From the
resampled data it recomputes the errors and the fitted order
$\hat q^*_b$, $b = 1, \dots, B$. Two intervals are available:

Percentile (default)
: The $\alpha/2$ and $1 - \alpha/2$ quantiles of $\hat q^*_1, \dots, \hat q^*_B$.

BCa (bias-corrected and accelerated, `method="bca"`)
: The quantiles at the adjusted levels
  $\alpha_{1,2} = \Phi\big(\hat z_0 + (\hat z_0 + z)/(1 - \hat a(\hat z_0 + z))\big)$
  with $z = z_{\alpha/2}$ and $z = z_{1-\alpha/2}$. The bias correction
  $\hat z_0 = \Phi^{-1}\big(\#\{\hat q^*_b < \hat q\}/B\big)$ shifts the
  interval when the estimate is not the median of the replicates; the
  acceleration $\hat a$ is estimated from the jackknife values
  $\hat q_{(i)}$ (the fit without path $i$),
  $\hat a = \sum_i (\hat q_{(\cdot)} - \hat q_{(i)})^3 \big/
  \big(6 \big(\sum_i (\hat q_{(\cdot)} - \hat q_{(i)})^2\big)^{3/2}\big)$.
  The jackknife values are cheap here: the leave-one-out means are
  $(n \bar Y_k - Y_{ik})/(n-1)$.

Both methods are described by Efron (1987) and Efron and Tibshirani (1993) †.
On identical replicates they agree with `scipy.stats.bootstrap` to rounding
error.

### How well do the intervals cover?

In theory, BCa intervals are second-order accurate and percentile intervals
first-order accurate. For the fitted order of a convergence study, simulations
show something else. The table gives the fraction of 95 % intervals that
contain the true order:

| Data | BCa | Percentile |
| --- | --- | --- |
| synthetic, mildly skewed, $n = 500$ | 94.7 % | 94.6 % |
| synthetic, strongly skewed, $n = 500$ | 89.8 % | 93.1 % |
| synthetic, extremely skewed, $n = 200$ | 84.4 % | 91.7 % |
| GBM, $\mu = 1$, $\sigma = 1$, Euler–Maruyama, $n = 1000$ | 81.2 % | 82.7 % |
| GBM, $\mu = 1$, $\sigma = 0.5$ | 91.9 % | 93.2 % |
| linear system (default parameters) | 91.4 % | 92.8 % |

The synthetic data are $Y_{ik} = h_k^{pq} Z_{ik}$ with
$Z_{ik} = \exp\big(s(\rho G_i + \sqrt{1-\rho^2}\, G_{ik}) - s^2/2\big)$,
$\rho = 0.8$, $s = 0.5, 1.5, 2$ and standard normal $G$, so the true order $q$
is known exactly; 4000 studies each, $B = 1000$. For the benchmark problems the
true order at the chosen step sizes ($h = 2^{-7}, \dots, 2^{-4}$) was fitted on
$10^6$ paths and compared with 1000 independent studies of $1000$ paths each.
The Monte Carlo error of the coverage is about one percentage point.

Two conclusions:

1. BCa was never better than the percentile interval and clearly worse for
   skewed data, presumably because the jackknife estimate of the acceleration
   is unstable when the pathwise errors have heavy tails. The percentile
   interval is therefore the default.
2. With heavy-tailed pathwise errors, as for geometric Brownian motion with
   $\sigma = 1$, both intervals cover noticeably less than the nominal level.
   Treat them as approximate; more paths help, and comparing the intervals of
   independent studies is a useful check.

The interval refers to the slope over the chosen step sizes, not to the
asymptotic order. When the error is not yet in the asymptotic range, a narrow
interval may exclude the theoretical order: for geometric Brownian motion with
the default parameters and $h = 2^{-9}, \dots, 2^{-4}$, 5000 paths give
$\hat q = 0.468$ with the interval $[0.461, 0.474]$.

## References

Efron, B. (1987). Better bootstrap confidence intervals. *J. Amer. Statist.
Assoc.* 82(397), 171–185 †. Efron, B. and Tibshirani, R. J. (1993). *An
Introduction to the Bootstrap.* Chapman & Hall, New York †.

† Details not yet checked against the publisher; the entries move to the
bibliography once they are.
