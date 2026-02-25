"""Tests for run_anuga.callbacks — callback protocol and implementations."""

import logging
import warnings

from run_anuga.callbacks import (
    HydrataCallback,
    LoggingCallback,
    NullCallback,
    SimulationCallback,
)
from run_anuga.logging_setup import (
    configure_simulation_logging,
    teardown_simulation_logging,
)


class TestNullCallback:
    def test_implements_protocol(self):
        assert isinstance(NullCallback(), SimulationCallback)

    def test_on_status_does_nothing(self):
        cb = NullCallback()
        cb.on_status("building mesh")  # should not raise

    def test_on_metric_does_nothing(self):
        cb = NullCallback()
        cb.on_metric("mesh_triangle_count", 42)

    def test_on_file_does_nothing(self):
        cb = NullCallback()
        cb.on_file("video", "/tmp/test.mp4")


class TestLoggingCallback:
    def test_implements_protocol(self):
        assert isinstance(LoggingCallback(), SimulationCallback)

    def test_on_status_logs_non_percentage(self, caplog):
        cb = LoggingCallback()
        with caplog.at_level(logging.INFO):
            cb.on_status("building mesh")
        assert "building mesh" in caplog.text

    def test_on_status_suppresses_percentage(self, caplog):
        cb = LoggingCallback()
        with caplog.at_level(logging.INFO):
            cb.on_status("45.2%")
        assert "45.2%" not in caplog.text

    def test_on_metric_logs(self, caplog):
        cb = LoggingCallback()
        with caplog.at_level(logging.INFO):
            cb.on_metric("memory_used", 1024)
        assert "memory_used" in caplog.text
        assert "1024" in caplog.text

    def test_on_file_logs(self, caplog):
        cb = LoggingCallback()
        with caplog.at_level(logging.INFO):
            cb.on_file("video", "/tmp/depth.mp4")
        assert "video" in caplog.text
        assert "/tmp/depth.mp4" in caplog.text

    def test_custom_logger(self):
        custom = logging.getLogger("test_custom")
        cb = LoggingCallback(logger_instance=custom)
        assert cb._logger is custom


class TestHydrataCallback:
    def test_implements_protocol(self):
        cb = HydrataCallback(
            username="u", password="p",
            control_server="https://test.com",
            project=1, scenario=2, run_id=3,
        )
        assert isinstance(cb, SimulationCallback)

    def test_url_property(self):
        cb = HydrataCallback(
            username="u", password="p",
            control_server="https://hydrata.com/",
            project=42, scenario=7, run_id=3,
        )
        assert cb._url == "https://hydrata.com/anuga/api/42/7/run/3/"

    def test_from_config(self):
        config = {
            "project": 10,
            "id": 5,
            "run_id": 2,
            "control_server": "https://example.com",
        }
        cb = HydrataCallback.from_config("user", "pass", config)
        assert cb.project == 10
        assert cb.scenario == 5
        assert cb.run_id == 2
        assert cb.control_server == "https://example.com"

    def test_from_config_defaults(self):
        cb = HydrataCallback.from_config("user", "pass", {})
        assert cb.project == 0
        assert cb.scenario == 0
        assert cb.run_id == 0
        assert cb.control_server == ""


class TestLoggingCallbackIntegration:
    """LoggingCallback + configure_simulation_logging integration."""

    def test_callback_messages_appear_in_log_file(self, tmp_path):
        sim_logger = configure_simulation_logging(str(tmp_path))
        try:
            cb = LoggingCallback(logger_instance=sim_logger)
            cb.on_status("45.2%")    # percentage — suppressed
            cb.on_status("building mesh")  # non-percentage — logged
            cb.on_metric("memory_used", 1024)

            log_file = tmp_path / "run_anuga_1.log"
            contents = log_file.read_text()
            assert "45.2%" not in contents        # percentage suppressed
            assert "building mesh" in contents    # non-percentage logged
            assert "memory_used" in contents
            assert "1024" in contents
        finally:
            teardown_simulation_logging()


class TestSetupLoggerDeprecation:
    def test_warns_deprecated(self, tmp_path):
        from run_anuga.run_utils import setup_logger

        input_data = {"output_directory": str(tmp_path), "scenario_config": {}}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            setup_logger(input_data)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
        teardown_simulation_logging()
