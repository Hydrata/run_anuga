"""Tests for run_anuga.logging_setup — simulation logging configuration."""

import logging
import sys
from unittest import mock

import pytest

from run_anuga.logging_setup import (
    _HANDLER_TAG,
    configure_simulation_logging,
    neutralize_anuga_logging,
    teardown_simulation_logging,
)


@pytest.fixture(autouse=True)
def _clean_root_logger():
    """Ensure every test starts and ends with a clean root logger."""
    root = logging.getLogger()
    original_level = root.level
    original_handlers = root.handlers[:]
    yield
    # Teardown: remove any tagged handlers and restore level.
    teardown_simulation_logging()
    root.handlers = original_handlers
    root.setLevel(original_level)


class TestConfigureSimulationLogging:
    def test_creates_log_file(self, tmp_path):
        logger = configure_simulation_logging(str(tmp_path))
        logger.info("hello")
        log_file = tmp_path / "run_anuga_1.log"
        assert log_file.exists()
        assert "hello" in log_file.read_text()

    def test_batch_number_in_filename(self, tmp_path):
        configure_simulation_logging(str(tmp_path), batch_number=3)
        logging.getLogger().info("batch test")
        log_file = tmp_path / "run_anuga_3.log"
        assert log_file.exists()
        assert "batch test" in log_file.read_text()

    def test_captures_root_logger_messages(self, tmp_path):
        """Messages on the root logger (simulates ANUGA) appear in the file."""
        configure_simulation_logging(str(tmp_path))
        logging.getLogger().critical("ANUGA timestep output")
        log_file = tmp_path / "run_anuga_1.log"
        assert "ANUGA timestep output" in log_file.read_text()

    def test_captures_named_logger_via_propagation(self, tmp_path):
        """Named logger messages propagate to root and appear in the file."""
        configure_simulation_logging(str(tmp_path))
        child = logging.getLogger("run_anuga.run_utils")
        child.info("child message")
        log_file = tmp_path / "run_anuga_1.log"
        assert "child message" in log_file.read_text()

    def test_returns_named_logger(self, tmp_path):
        logger = configure_simulation_logging(str(tmp_path))
        assert logger.name == "run_anuga.sim"

    def test_no_duplicate_handlers_on_repeated_calls(self, tmp_path):
        configure_simulation_logging(str(tmp_path))
        configure_simulation_logging(str(tmp_path))
        root = logging.getLogger()
        tagged = [h for h in root.handlers if getattr(h, _HANDLER_TAG, False)]
        # Should have exactly 2 tagged handlers (file + console) not 4.
        assert len(tagged) == 2

    def test_output_dir_created_if_missing(self, tmp_path):
        new_dir = tmp_path / "nested" / "output"
        configure_simulation_logging(str(new_dir))
        assert new_dir.exists()


class TestTeardown:
    def test_removes_tagged_handlers(self, tmp_path):
        configure_simulation_logging(str(tmp_path))
        root = logging.getLogger()
        assert any(getattr(h, _HANDLER_TAG, False) for h in root.handlers)
        teardown_simulation_logging()
        assert not any(getattr(h, _HANDLER_TAG, False) for h in root.handlers)

    def test_preserves_non_tagged_handlers(self, tmp_path):
        foreign = logging.StreamHandler()
        root = logging.getLogger()
        root.addHandler(foreign)
        configure_simulation_logging(str(tmp_path))
        teardown_simulation_logging()
        assert foreign in root.handlers
        root.removeHandler(foreign)

    def test_safe_to_call_twice(self, tmp_path):
        configure_simulation_logging(str(tmp_path))
        teardown_simulation_logging()
        teardown_simulation_logging()  # Should not raise.

    def test_restores_root_level(self, tmp_path):
        root = logging.getLogger()
        root.setLevel(logging.WARNING)
        configure_simulation_logging(str(tmp_path))
        assert root.level != logging.WARNING  # Changed by configure
        teardown_simulation_logging()
        assert root.level == logging.WARNING


class TestNeutralizeAnugaLogging:
    def test_no_error_without_anuga(self, tmp_path):
        """Safe no-op when anuga is not installed."""
        # Temporarily make anuga un-importable
        with mock.patch.dict(sys.modules, {"anuga": None, "anuga.utilities": None, "anuga.utilities.log": None}):
            neutralize_anuga_logging(str(tmp_path))  # Should not raise.

    def test_sets_setup_flag(self, tmp_path):
        """When anuga.utilities.log is importable, _setup is set to True."""
        fake_log = mock.MagicMock()
        fake_log._setup = False
        fake_log.log_filename = ""
        fake_log.console_logging_level = logging.DEBUG

        # Wire up the module hierarchy so `import anuga.utilities.log` resolves to fake_log.
        fake_anuga = mock.MagicMock()
        fake_utilities = mock.MagicMock()
        fake_anuga.utilities = fake_utilities
        fake_utilities.log = fake_log

        with mock.patch.dict(sys.modules, {
            "anuga": fake_anuga,
            "anuga.utilities": fake_utilities,
            "anuga.utilities.log": fake_log,
        }):
            neutralize_anuga_logging(str(tmp_path))

        assert fake_log._setup is True
        assert "anuga_internal.log" in fake_log.log_filename
        assert fake_log.console_logging_level > logging.CRITICAL


class TestDjangoMode:
    def test_no_console_handler_in_django_mode(self, tmp_path):
        """In Django mode, no StreamHandler is added."""
        with mock.patch(
            "run_anuga.logging_setup._is_django_configured", return_value=True
        ):
            configure_simulation_logging(str(tmp_path))
        root = logging.getLogger()
        tagged = [h for h in root.handlers if getattr(h, _HANDLER_TAG, False)]
        assert len(tagged) == 1  # Only the FileHandler
        assert isinstance(tagged[0], logging.FileHandler)

    def test_does_not_change_root_level_in_django_mode(self, tmp_path):
        root = logging.getLogger()
        root.setLevel(logging.WARNING)
        with mock.patch(
            "run_anuga.logging_setup._is_django_configured", return_value=True
        ):
            configure_simulation_logging(str(tmp_path))
        assert root.level == logging.WARNING
