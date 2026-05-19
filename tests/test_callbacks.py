"""Tests for run_anuga.callbacks — callback protocol and implementations."""

import logging
from unittest.mock import MagicMock, patch

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


class TestHydrataCallbackOnProgress:
    """W6 (TASK-1044) — HydrataCallback.on_progress POSTs to V2 /progress/."""

    def _make_cb(self):
        return HydrataCallback(
            username="anuga_admin",
            password="p",
            control_server="https://hydrata.com/",
            project=42,
            scenario=7,
            run_id=99,
        )

    def test_v2_progress_url(self):
        cb = self._make_cb()
        assert cb._v2_progress_url == "https://hydrata.com/api/v2/anuga/runs/99/progress/"

    def test_on_progress_posts_to_v2_progress(self):
        """Calling on_progress hits the V2 progress URL with the right body."""
        cb = self._make_cb()

        with patch("run_anuga._imports.import_optional") as mock_import:
            mock_requests = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_requests.Session.return_value.post.return_value = mock_response
            mock_import.return_value = mock_requests

            cb.on_progress(42.0, eta_seconds=300)

        call = mock_requests.Session.return_value.post.call_args
        # URL is the first positional arg
        assert call.args[0] == "https://hydrata.com/api/v2/anuga/runs/99/progress/"
        # Body via json kwarg
        body = call.kwargs.get('json') or {}
        assert body['progress_pct'] == 42.0
        assert body['eta_seconds'] == 300

    def test_on_progress_eta_none_serialises_as_null(self):
        """eta_seconds=None must be sent as JSON null, not 0 or omitted as '0'."""
        cb = self._make_cb()

        with patch("run_anuga._imports.import_optional") as mock_import:
            mock_requests = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_requests.Session.return_value.post.return_value = mock_response
            mock_import.return_value = mock_requests

            cb.on_progress(0.0, eta_seconds=None)

        call = mock_requests.Session.return_value.post.call_args
        body = call.kwargs.get('json') or {}
        assert body['progress_pct'] == 0.0
        assert body['eta_seconds'] is None

    def test_on_progress_swallows_exceptions(self):
        """A request blowup must not propagate (no-op on network failure)."""
        cb = self._make_cb()

        with patch("run_anuga._imports.import_optional") as mock_import:
            mock_import.side_effect = RuntimeError("requests not installed")
            # MUST NOT raise — logger.exception is the safety net
            cb.on_progress(42.0, eta_seconds=300)
