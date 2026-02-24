"""Tests for HydrataCallback HTTP edge cases and error handling."""

from unittest.mock import MagicMock, patch

import pytest

from run_anuga.callbacks import HydrataCallback


class TestHydrataCallbackPatch:
    @patch("run_anuga._imports.import_optional")
    def test_http_error_logged_not_raised(self, mock_import, caplog):
        """HTTP errors are logged but don't raise exceptions."""
        mock_requests = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_requests.Session.return_value.patch.return_value = mock_response
        mock_requests.auth.HTTPBasicAuth.return_value = ("user", "pass")
        mock_import.return_value = mock_requests

        cb = HydrataCallback("user", "pass", "http://example.com/", 1, 1, 1)
        import logging
        with caplog.at_level(logging.ERROR):
            cb.on_status("running")
        assert "500" in caplog.text
        assert "Error updating web interface" in caplog.text
        assert "Internal Server Error" in caplog.text

    @patch("run_anuga._imports.import_optional")
    def test_http_success_no_error_log(self, mock_import, caplog):
        """Successful HTTP calls don't log errors."""
        mock_requests = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "OK"
        mock_requests.Session.return_value.patch.return_value = mock_response
        mock_requests.auth.HTTPBasicAuth.return_value = ("user", "pass")
        mock_import.return_value = mock_requests

        cb = HydrataCallback("user", "pass", "http://example.com/", 1, 1, 1)
        import logging
        with caplog.at_level(logging.ERROR):
            cb.on_status("running")
        assert "Error" not in caplog.text

    @patch("run_anuga._imports.import_optional")
    def test_on_metric_sends_key_value(self, mock_import):
        """on_metric sends the key-value pair as PATCH data."""
        mock_requests = MagicMock()
        mock_response = MagicMock(status_code=200)
        mock_session = MagicMock()
        mock_session.patch.return_value = mock_response
        mock_requests.Session.return_value = mock_session
        mock_requests.auth.HTTPBasicAuth.return_value = ("user", "pass")
        mock_import.return_value = mock_requests

        cb = HydrataCallback("user", "pass", "http://example.com/", 1, 2, 3)
        cb.on_metric("memory_used", 1024)

        call_args = mock_session.patch.call_args
        data = call_args.kwargs.get("data", {})
        assert data["memory_used"] == 1024
        # _patch() always injects project and scenario into every request
        assert data["project"] == 1
        assert data["scenario"] == 2


class TestHydrataCallbackOnFile:
    def test_on_file_nonexistent_raises(self):
        """on_file with nonexistent file raises FileNotFoundError."""
        cb = HydrataCallback("user", "pass", "http://example.com/", 1, 1, 1)
        with pytest.raises(FileNotFoundError):
            cb.on_file("result", "/nonexistent/file.tif")

    @patch("run_anuga._imports.import_optional")
    def test_on_file_opens_file(self, mock_import, tmp_path):
        """on_file opens the file and sends it."""
        mock_requests = MagicMock()
        mock_response = MagicMock(status_code=200)
        mock_session = MagicMock()
        mock_session.patch.return_value = mock_response
        mock_requests.Session.return_value = mock_session
        mock_requests.auth.HTTPBasicAuth.return_value = ("user", "pass")
        mock_import.return_value = mock_requests

        test_file = tmp_path / "result.tif"
        test_file.write_bytes(b"fake tif data")

        cb = HydrataCallback("user", "pass", "http://example.com/", 1, 1, 1)
        cb.on_file("result", str(test_file))

        call_args = mock_session.patch.call_args
        files = call_args.kwargs.get("files")
        assert files is not None
        assert "result" in files


class TestHydrataCallbackURLEdgeCases:
    def test_url_trailing_slash(self):
        cb = HydrataCallback("u", "p", "http://example.com/", 1, 2, 3)
        assert cb._url == "http://example.com/anuga/api/1/2/run/3/"

    def test_url_no_trailing_slash(self):
        cb = HydrataCallback("u", "p", "http://example.com", 1, 2, 3)
        assert cb._url == "http://example.com/anuga/api/1/2/run/3/"

    def test_url_with_path(self):
        cb = HydrataCallback("u", "p", "https://hydrata.com/", 5, 10, 42)
        assert cb._url == "https://hydrata.com/anuga/api/5/10/run/42/"

    def test_from_config_empty_dict(self):
        cb = HydrataCallback.from_config("user", "pass", {})
        assert cb.control_server == ""
        assert cb.project == 0
        assert cb.scenario == 0
        assert cb.run_id == 0
