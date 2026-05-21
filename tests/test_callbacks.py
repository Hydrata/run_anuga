"""Tests for run_anuga.callbacks — callback protocol and implementations.

TASK-1049 (W1 of TASK-1048): HydrataCallback was rewritten to use the V2
API + ``X-Internal-Token`` header + owned ``requests.Session``. The
V1-era TestHydrataCallback / TestHydrataCallbackOnProgress classes were
removed because they exercised the removed BasicAuth surface and V1
``/anuga/api/<p>/<s>/run/<r>/`` URL. New coverage lives in
``test_callbacks_v2.py``.
"""

import logging

from run_anuga.callbacks import (
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

    def test_close_is_noop(self):
        """TASK-1049: NullCallback exposes close() so run_sim can always call it."""
        cb = NullCallback()
        cb.close()  # must not raise


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

    def test_close_is_noop(self):
        """TASK-1049: LoggingCallback exposes close() so run_sim can always call it."""
        cb = LoggingCallback()
        cb.close()  # must not raise


class TestNullCallbackOnProgress:
    """W6 (TASK-1044) — NullCallback.on_progress is a silent no-op."""

    def test_on_progress_does_nothing(self):
        cb = NullCallback()
        cb.on_progress(50.0, eta_seconds=300)  # should not raise

    def test_on_progress_eta_none(self):
        cb = NullCallback()
        cb.on_progress(0.0, eta_seconds=None)  # should not raise


class TestLoggingCallbackOnProgress:
    """W6 (TASK-1044) — LoggingCallback.on_progress logs pct + eta via logging."""

    def test_on_progress_logs(self, caplog):
        cb = LoggingCallback()
        with caplog.at_level(logging.INFO):
            cb.on_progress(42.5, eta_seconds=300)
        assert '42.5' in caplog.text
        assert '300' in caplog.text

    def test_on_progress_eta_none_logs(self, caplog):
        cb = LoggingCallback()
        with caplog.at_level(logging.INFO):
            cb.on_progress(0.0, eta_seconds=None)
        assert '0.0' in caplog.text
