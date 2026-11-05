"""Empirical convergence studies and their statistics.

:class:`StrongConvergenceStudy` estimates strong errors of schemes on one
Brownian path and fits the order of convergence with a pathwise bootstrap
interval; :mod:`numsde.analysis.statistics` holds the underlying estimators.
"""

from numsde.analysis import statistics
from numsde.analysis.convergence import StrongConvergenceResult, StrongConvergenceStudy

__all__ = ["StrongConvergenceResult", "StrongConvergenceStudy", "statistics"]
