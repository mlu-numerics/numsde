import importlib
import sys

import numpy as np
import pytest

import numsde
from numsde._backend import SUPPORTED_BACKENDS, array_namespace, namespace_for


def test_default_namespace_is_numpy():
    assert array_namespace().__name__ == "array_api_compat.numpy"


@pytest.mark.parametrize("scalar", [None, 1, 2.5, 1 + 2j, True, np.float64(3.0)])
def test_scalars_and_none_are_ignored(scalar):
    xp = array_namespace(np.zeros(2), scalar)
    assert xp.__name__ == "array_api_compat.numpy"


def test_only_scalars_fall_back_to_numpy():
    assert array_namespace(1.0, None).__name__ == "array_api_compat.numpy"


def test_numpy_array_resolves_to_numpy():
    xp = array_namespace(np.ones((2, 3)))
    assert xp is namespace_for("numpy")


def test_non_array_raises_type_error():
    with pytest.raises(TypeError):
        array_namespace([1.0, 2.0])


def test_unknown_backend_raises_value_error():
    with pytest.raises(ValueError, match="Unknown backend 'jax'"):
        namespace_for("jax")


def test_missing_backend_gives_install_hint(monkeypatch):
    def fail(name):
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", fail)
    with pytest.raises(ImportError, match=r"numsde\[torch\]"):
        namespace_for("torch")


def test_supported_backends():
    assert SUPPORTED_BACKENDS == ("numpy", "torch")


def test_torch_backend_when_installed():
    pytest.importorskip("torch")
    xp = namespace_for("torch")
    assert xp.__name__ == "array_api_compat.torch"
    assert array_namespace(xp.ones(2)) is xp


def test_public_api():
    assert numsde.array_namespace is array_namespace
    assert isinstance(numsde.__version__, str)


def test_import_does_not_require_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` fail
    for name in [m for m in sys.modules if m.startswith("numsde")]:
        monkeypatch.delitem(sys.modules, name)
    reloaded = importlib.import_module("numsde")
    assert reloaded.array_namespace().__name__ == "array_api_compat.numpy"
