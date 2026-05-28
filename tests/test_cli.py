"""Tests for run_anuga.cli — CLI subcommands."""

import os
import subprocess
import sys
from unittest import mock


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
        for cmd in ["run", "run-and-report", "validate", "info", "post-process", "viz", "upload"]:
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


class TestRunSubcommandCallbackSelection:
    """TASK-1160 (F1b): the `run` subcommand must not force LoggingCallback.

    Bare ``python run.py`` already auto-constructs HydrataCallback when
    HYDRATA_INTERNAL_COMPUTE_TOKEN + scenario_config.control_server are
    present; the CLI must match that. Pass ``--log-to-stdout`` to force
    LoggingCallback for standalone debugging.
    """

    def _invoke_cmd_run(self, log_to_stdout: bool):
        """Drive cmd_run with run_sim mocked out, return the kwargs it received."""
        from run_anuga import cli

        args = mock.MagicMock(
            username=None,
            password=None,
            package_dir="/tmp/fake",
            batch_number=1,
            checkpoint_time=None,
            log_to_stdout=log_to_stdout,
        )
        with mock.patch("run_anuga.run.run_sim") as mock_run_sim:
            cli.cmd_run(args)
        return mock_run_sim.call_args.kwargs

    def test_default_callback_is_none(self):
        """Default callback is None so run_sim's env-based auto-construct kicks in."""
        kwargs = self._invoke_cmd_run(log_to_stdout=False)
        assert kwargs["callback"] is None

    def test_log_to_stdout_forces_logging_callback(self):
        """--log-to-stdout passes a LoggingCallback for silent standalone debugging."""
        from run_anuga.callbacks import LoggingCallback

        kwargs = self._invoke_cmd_run(log_to_stdout=True)
        assert isinstance(kwargs["callback"], LoggingCallback)


class TestRunAndReportSubcommand:
    """TASK-1159 (F1): the new run-and-report subcommand exists + wires through."""

    def test_subcommand_in_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "run_anuga.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "run-and-report" in result.stdout

    def test_calls_run_and_report_with_package_dir_and_bucket(self):
        from run_anuga import cli

        args = mock.MagicMock(package_dir="/tmp/fake", result_bucket="my-bucket")
        with mock.patch("run_anuga._handoff.run_and_report") as mock_rar:
            mock_rar.return_value = {"result_key": "1_2_3_results.zip", "process_result_status": 202}
            cli.cmd_run_and_report(args)
        mock_rar.assert_called_once_with("/tmp/fake", result_bucket="my-bucket")
