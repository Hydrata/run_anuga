"""Tests for cli.py:resolve_package_dir() edge cases."""

import argparse
import os

import pytest

from run_anuga.cli import resolve_package_dir


class TestResolvePackageDir:
    def test_with_scenario_json_file(self, tmp_path):
        (tmp_path / "scenario.json").write_text("{}")
        result = resolve_package_dir(str(tmp_path / "scenario.json"))
        assert result == str(tmp_path)

    def test_with_directory(self, tmp_path):
        result = resolve_package_dir(str(tmp_path))
        assert result == str(tmp_path)

    def test_rejects_non_scenario_file(self, tmp_path):
        (tmp_path / "other.json").write_text("{}")
        with pytest.raises(argparse.ArgumentTypeError, match="Expected scenario.json"):
            resolve_package_dir(str(tmp_path / "other.json"))

    def test_nonexistent_path(self):
        with pytest.raises(argparse.ArgumentTypeError, match="does not exist"):
            resolve_package_dir("/nonexistent/path/abc123")

    def test_returns_absolute_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = resolve_package_dir(".")
        assert os.path.isabs(result)

    def test_directory_without_scenario_json(self, tmp_path):
        """Directory without scenario.json is still accepted (validation is separate)."""
        result = resolve_package_dir(str(tmp_path))
        assert result == str(tmp_path)

    def test_nested_scenario_json(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        nested.mkdir(parents=True)
        (nested / "scenario.json").write_text("{}")
        result = resolve_package_dir(str(nested / "scenario.json"))
        assert result == str(nested)
