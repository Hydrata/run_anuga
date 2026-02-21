"""Tests for run_anuga.callbacks — callback protocol and implementations."""

import logging

from run_anuga.callbacks import (
    HydrataCallback,
    LoggingCallback,
    NullCallback,
    SimulationCallback,
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

    def test_on_status_logs(self, caplog):
        cb = LoggingCallback()
        with caplog.at_level(logging.INFO):
            cb.on_status("45.2%")
        assert "45.2%" in caplog.text

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
