"""Tests for run_anuga._http.post_to_control_server.

Covers happy path (201), 4xx, 5xx, connection error, and the dispatch
between POST and PATCH.  Uses unittest.mock to match the existing
test-suite style (responses lib is available but not used elsewhere).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
import requests
from requests.auth import HTTPBasicAuth

from run_anuga._http import post_to_control_server


@pytest.fixture
def auth():
    return HTTPBasicAuth("user", "pass")


def _make_session_mock(status_code: int, text: str = "ok"):
    """Build a MagicMock session.  Returns (session, fake_response)."""
    fake_response = MagicMock(spec=requests.Response)
    fake_response.status_code = status_code
    fake_response.text = text

    session = MagicMock()
    session.__enter__.return_value = session
    session.__exit__.return_value = None
    session.post.return_value = fake_response
    session.patch.return_value = fake_response
    return session, fake_response


class TestHappyPath:
    def test_post_201_returns_response_no_error_log(self, auth, caplog):
        session, fake_response = _make_session_mock(201)
        with patch("requests.Session", return_value=session), caplog.at_level(logging.ERROR):
            result = post_to_control_server(
                "https://example.com/api/v2/anuga/runs/1/error/",
                auth=auth,
                method="POST",
                data={"message": "boom"},
            )

        assert result is fake_response
        # Default timeout is None (no timeout) to preserve pre-refactor
        # behavior and avoid breaking PATCH-with-files callers on slow links.
        session.post.assert_called_once_with(
            "https://example.com/api/v2/anuga/runs/1/error/",
            data={"message": "boom"},
            files=None,
            timeout=None,
        )
        assert session.auth is auth
        assert "Error posting to control server" not in caplog.text

    def test_patch_200_returns_response_no_error_log(self, auth, caplog):
        session, fake_response = _make_session_mock(200)
        with patch("requests.Session", return_value=session), caplog.at_level(logging.ERROR):
            result = post_to_control_server(
                "https://example.com/anuga/api/1/2/run/3/",
                auth=auth,
                method="PATCH",
                data={"status": "running"},
                files={"f": b"x"},
            )

        assert result is fake_response
        # PATCH-with-files must default to no timeout (mesh/result artifact
        # uploads from a worker on a slow link can exceed any short bound).
        session.patch.assert_called_once_with(
            "https://example.com/anuga/api/1/2/run/3/",
            data={"status": "running"},
            files={"f": b"x"},
            timeout=None,
        )
        assert "Error posting to control server" not in caplog.text

    def test_method_is_case_insensitive(self, auth):
        session, _ = _make_session_mock(200)
        with patch("requests.Session", return_value=session):
            post_to_control_server("https://example.com/", auth=auth, method="post")
        session.post.assert_called_once()

    def test_session_auth_set_on_session_not_per_request(self, auth):
        session, _ = _make_session_mock(200)
        with patch("requests.Session", return_value=session):
            post_to_control_server("https://example.com/", auth=auth, method="POST")
        # auth is set on session attribute, not passed as kwarg to .post
        assert session.auth is auth
        _args, kwargs = session.post.call_args
        assert "auth" not in kwargs


class TestTimeout:
    def test_default_timeout_is_none(self, auth):
        """Helper default must be None (no timeout) to match pre-refactor
        behavior. PATCH-with-files callers (mesh/result artifact uploads)
        depend on this — a 30s default was a regression that raised
        requests.exceptions.Timeout mid-upload on slow links.
        """
        session, _ = _make_session_mock(200)
        with patch("requests.Session", return_value=session):
            post_to_control_server("https://example.com/", auth=auth, method="POST")
        _args, kwargs = session.post.call_args
        assert kwargs["timeout"] is None

    def test_default_timeout_is_none_for_patch(self, auth):
        session, _ = _make_session_mock(200)
        with patch("requests.Session", return_value=session):
            post_to_control_server("https://example.com/", auth=auth, method="PATCH")
        _args, kwargs = session.patch.call_args
        assert kwargs["timeout"] is None

    def test_explicit_timeout_honored_post(self, auth):
        """Callers (e.g. _report_run_error) can pass an explicit bound."""
        session, _ = _make_session_mock(200)
        with patch("requests.Session", return_value=session):
            post_to_control_server(
                "https://example.com/", auth=auth, method="POST", timeout=30,
            )
        _args, kwargs = session.post.call_args
        assert kwargs["timeout"] == 30

    def test_explicit_timeout_honored_patch(self, auth):
        session, _ = _make_session_mock(200)
        with patch("requests.Session", return_value=session):
            post_to_control_server(
                "https://example.com/", auth=auth, method="PATCH", timeout=5,
            )
        _args, kwargs = session.patch.call_args
        assert kwargs["timeout"] == 5


class TestErrorStatusLogging:
    def test_4xx_logs_error_returns_response(self, auth, caplog):
        session, fake_response = _make_session_mock(403, text="forbidden")
        url = "https://example.com/api/v2/anuga/runs/1/error/"
        with patch("requests.Session", return_value=session), caplog.at_level(logging.ERROR):
            result = post_to_control_server(url, auth=auth, method="POST")

        assert result is fake_response
        assert "Error posting to control server" in caplog.text
        assert "403" in caplog.text
        assert "forbidden" in caplog.text
        # URL is included in the log line so distinct call-sites are
        # distinguishable in production logs.
        assert url in caplog.text

    def test_5xx_logs_error_returns_response(self, auth, caplog):
        session, fake_response = _make_session_mock(503, text="unavailable")
        url = "https://example.com/anuga/api/1/2/run/3/"
        with patch("requests.Session", return_value=session), caplog.at_level(logging.ERROR):
            result = post_to_control_server(url, auth=auth, method="PATCH")

        assert result is fake_response
        assert "Error posting to control server" in caplog.text
        assert "503" in caplog.text
        assert "unavailable" in caplog.text
        assert url in caplog.text


class TestConnectionError:
    def test_connection_error_propagates(self, auth):
        session = MagicMock()
        session.__enter__.return_value = session
        session.__exit__.return_value = None
        session.post.side_effect = requests.exceptions.ConnectionError("dns fail")

        with patch("requests.Session", return_value=session):
            with pytest.raises(requests.exceptions.ConnectionError):
                post_to_control_server("https://example.com/", auth=auth, method="POST")

    def test_timeout_propagates(self, auth):
        session = MagicMock()
        session.__enter__.return_value = session
        session.__exit__.return_value = None
        session.patch.side_effect = requests.exceptions.Timeout("too slow")

        with patch("requests.Session", return_value=session):
            with pytest.raises(requests.exceptions.Timeout):
                post_to_control_server("https://example.com/", auth=auth, method="PATCH")


class TestInvalidMethod:
    def test_unsupported_method_raises_value_error(self, auth):
        session, _ = _make_session_mock(200)
        with patch("requests.Session", return_value=session):
            with pytest.raises(ValueError, match="Unsupported HTTP method"):
                post_to_control_server("https://example.com/", auth=auth, method="DELETE")
