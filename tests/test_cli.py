"""Tests for run_anuga.cli — CLI subcommands."""

import os
import subprocess
import sys


FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "data", "minimal_package")


class TestMainHelp:
    def test_no_args_prints_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "run_anuga.cli"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "run-anuga" in result.stderr or "run-anuga" in result.stdout

    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "run_anuga.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        for cmd in ["run", "validate", "info", "post-process", "viz", "upload"]:
            assert cmd in result.stdout


class TestValidateSubcommand:
    def test_valid_package(self):
        result = subprocess.run(
            [sys.executable, "-m", "run_anuga.cli", "validate", FIXTURE_DIR],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Valid scenario: run_1_1_1" in result.stdout
        assert "Duration: 600s" in result.stdout

    def test_invalid_package(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "run_anuga.cli", "validate", str(tmp_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Invalid" in result.stderr


class TestInfoSubcommand:
    def test_info_valid_package(self):
        result = subprocess.run(
            [sys.executable, "-m", "run_anuga.cli", "info", FIXTURE_DIR],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Label:   run_1_1_1" in result.stdout
        assert "EPSG:    EPSG:28355" in result.stdout
        assert "Duration: 600s" in result.stdout
        assert "boundary.geojson" in result.stdout


class TestRunSubcommandImport:
    def test_run_import_error_mentions_extra(self):
        """run subcommand should fail with ImportError mentioning pip install."""
        result = subprocess.run(
            [sys.executable, "-m", "run_anuga.cli", "run", FIXTURE_DIR],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert 'pip install "run_anuga[full]"' in result.stderr

    def test_post_process_import_error(self):
        result = subprocess.run(
            [sys.executable, "-m", "run_anuga.cli", "post-process", FIXTURE_DIR],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "pip install" in result.stderr

    def test_viz_import_error(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "run_anuga.cli", "viz", str(tmp_path), "depth"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "pip install" in result.stderr


class TestOldMainStillImportable:
    def test_old_main_importable(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from run_anuga.run import main; print('ok')",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "ok" in result.stdout
