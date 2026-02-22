"""Tests for run_anuga._imports — lazy import helper."""

import pytest

from run_anuga._imports import import_optional


def test_import_stdlib_module():
    """Importing a stdlib module should succeed."""
    mod = import_optional("json")
    assert hasattr(mod, "loads")


def test_import_pydantic():
    """Pydantic is a core dep and should always be available."""
    mod = import_optional("pydantic")
    assert hasattr(mod, "BaseModel")


def test_import_nonexistent_raises():
    """A missing module should raise ImportError with install hint."""
    with pytest.raises(ImportError, match='pip install "run_anuga'):
        import_optional("nonexistent_module_xyz")


def test_import_hint_uses_extra_map():
    """The error message should reference the correct pip extra."""
    with pytest.raises(ImportError, match="full"):
        import_optional("anuga")


def test_import_hint_custom_extra():
    """Passing extra= explicitly should override the default lookup."""
    with pytest.raises(ImportError, match="myextra"):
        import_optional("nonexistent_module_xyz", extra="myextra")


def test_import_run_anuga_succeeds():
    """The top-level import should work without optional deps."""
    import run_anuga
    assert hasattr(run_anuga, "__version__")


def test_import_config_succeeds():
    """config module only needs pydantic (core dep)."""
    from run_anuga.config import ScenarioConfig
    assert hasattr(ScenarioConfig, "from_package")


def test_import_callbacks_succeeds():
    """callbacks module has no heavy deps."""
    from run_anuga.callbacks import NullCallback, LoggingCallback, SimulationCallback
    assert NullCallback is not None
    assert LoggingCallback is not None
    assert SimulationCallback is not None
