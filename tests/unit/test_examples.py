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
    table = example.main(["--n-paths", "400", "--k-max", "3", "--seed", "1"])
    np.testing.assert_array_equal(table.n_steps, [10, 20, 40])
    assert np.all(table.lower <= table.errors)
    assert np.all(table.errors <= table.upper)
    assert np.all(np.diff(table.errors) < 0)
    assert table.eoc.shape == (2,)
    output = capsys.readouterr().out
    assert "95% conf. interval" in output
    assert len(output.splitlines()) == 5  # header, rule and three rows
