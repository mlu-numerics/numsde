"""Backend layer: array namespaces and random number generation.

All numerical code in NumSDE is written against the `Python Array API
standard <https://data-apis.org/array-api/>`_. This subpackage resolves the
array namespace (NumPy, PyTorch, ...) to use and wraps random number
generation, which the standard does not cover.

Backend-specific code belongs here and nowhere else.
"""

from numsde._backend.arrays import is_real_floating, owned_read_only
from numsde._backend.namespace import (
    SUPPORTED_BACKENDS,
    array_namespace,
    namespace_for,
)
from numsde._backend.rng import (
    RandomNumberGenerator,
    SeedLike,
    default_rng,
    derive_seed_sequence,
    frozen_seed_sequence,
)

__all__ = [
    "SUPPORTED_BACKENDS",
    "RandomNumberGenerator",
    "SeedLike",
    "array_namespace",
    "default_rng",
    "derive_seed_sequence",
    "frozen_seed_sequence",
    "is_real_floating",
    "namespace_for",
    "owned_read_only",
]
