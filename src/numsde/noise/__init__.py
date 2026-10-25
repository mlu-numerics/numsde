"""Driving noise processes and sampled noise paths.

A noise path is a first-class object: it is sampled once, can be refined and
coarsened consistently, and can be reused across schemes and step sizes. This
is what makes convergence studies, adaptive step sizes and multilevel Monte
Carlo correct.
"""

from numsde.noise.base import NoiseProcess
from numsde.noise.brownian import BrownianPath, WienerProcess

__all__ = ["BrownianPath", "NoiseProcess", "WienerProcess"]
