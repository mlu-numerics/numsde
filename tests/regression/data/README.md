# Reference data from legacy scripts

Reference numbers from seeded NumPy scripts of the book project are stored
here once, together with seed, parameters and the SHA-256 checksums of the
script files that produced them. The scripts themselves stay with the book
project and are not part of this repository.

## `legacy_gbm_em_l2error.npz`

| Field | Content |
| --- | --- |
| Source | `num_exp_GBM.py`, function `compute_L2_errors`, with `StochODEs.py` and `SODESolvers.py` |
| Change to the script | none; only the arguments differ from `main()` |
| Solvers | `EulerMaruyama` only (Milstein follows in milestone M2) |
| Problem | geometric Brownian motion, `mu = 1/4`, `sigma = 1`, `X0 = 1`, `t_span = [0, 1]` |
| Seed | 20250718, as in `main()` |
| Sample size | 2000 paths instead of 50 000, to keep the test fast |
| Step counts | `N = 10 * 2**k` for `k = 0, ..., 5` instead of `k = 0, ..., 7` |
| Error | `L2_error()`: (E[max_i \|X(t_i) − X_h(t_i)\|²])^(1/2), the library's default strong error |
| NumPy | 2.5.4 |
| Generated | 2026-11-02 |

The test `tests/regression/test_legacy_gbm.py` rebuilds the legacy Brownian
paths and checks that the library reproduces the errors.

To regenerate the data, run the following with the legacy scripts on the
path (they need SciPy and matplotlib):

```python
import numpy as np
import matplotlib

matplotlib.use("Agg")
from num_exp_GBM import compute_L2_errors
from SODESolvers import EulerMaruyama
from StochODEs import GeometricBrownianMotion

GBM = GeometricBrownianMotion(
    t_span=[0.0, 1.0],
    num_samples=2000,
    seed=20250718,
    param={"mu": 1 / 4, "sigma": 1.0},
    X0=1,
)
N_list = np.array([10 * 2**k for k in range(6)])
errors = compute_L2_errors(N_list, GBM, [EulerMaruyama])[0]
```
