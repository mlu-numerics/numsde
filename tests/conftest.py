"""Shared test configuration."""

import os

from hypothesis import HealthCheck, settings

# Deterministic, CI-friendly Hypothesis profiles. Select with HYPOTHESIS_PROFILE.
settings.register_profile("ci", max_examples=200, derandomize=True, deadline=None)
settings.register_profile("dev", max_examples=50, deadline=None)
settings.register_profile(
    "thorough",
    max_examples=2000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
