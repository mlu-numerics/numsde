"""Smoke tests: the ported legacy scripts in ``examples/`` run."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).parents[2] / "examples"


def load(name):
    path = EXAMPLES / f"{name}.py"
    if not path.exists():  # pragma: no cover - only for the installed-wheel job
        pytest.skip(f"{path} is not available")
    spec = importlib.util.spec_from_file_location(f"example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strong_error_gbm_runs_and_prints_a_table(capsys):
    example = load("strong_error_gbm")
    result = example.main(["--n-paths", "400", "--k-max", "3", "--seed", "1"])
    np.testing.assert_allclose(result.step_sizes, [1 / 40, 1 / 20, 1 / 10])
    assert np.all(np.diff(result.errors[0]) > 0)
    assert result.orders_ci.shape == (1, 2)
    output = capsys.readouterr().out
    assert "Euler–Maruyama (expected order 0.5)" in output
    assert "fitted order" in output


def test_strong_error_gbm_plot(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    example = load("strong_error_gbm")
    example.main(["--n-paths", "100", "--k-max", "2", "--plot"])
