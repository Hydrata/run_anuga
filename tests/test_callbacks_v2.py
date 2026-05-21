"""Tests for HydrataCallback V2 migration (TASK-1049 / W1 of TASK-1048).

Covers:
* Owned ``requests.Session`` lifecycle (created in ``__init__``, released
  by ``close()`` — folds TASK-990 per AC7).
* V2 URL format ``/api/v2/anuga/runs/<id>/{log,progress}/``.
* ``X-Internal-Token`` header (RAW, no ``Bearer`` prefix).
* Absence of ``Authorization`` header.
* Fail-fast on missing ``HYDRATA_INTERNAL_COMPUTE_TOKEN`` env var.

The constructor signature dropped ``username``/``password`` (no BasicAuth);
the only ID kept in the URL is ``run_id`` (server infers project/scenario
from the run row).
"""

from __future__ import annotations

from unittest import mock

import pytest

from run_anuga.callbacks import HydrataCallback


@pytest.fixture
def token_env(monkeypatch):
    """Set the internal-token env var for tests that need a working callback."""
    monkeypatch.setenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', 'test-token-123')


def _make_cb():
    return HydrataCallback(
        control_server='https://hydrata.com/',
        project=42,
        scenario=7,
        run_id=99,
    )


def test_init_creates_session_once(token_env):
    """One ``requests.Session()`` per HydrataCallback instance (owned)."""
    with mock.patch('requests.Session') as mock_session_ctor:
        mock_session_ctor.return_value = mock.MagicMock(headers={})
        cb = _make_cb()
    assert mock_session_ctor.call_count == 1
    # The instance attribute is the Session returned by the patched ctor.
    assert cb.session is mock_session_ctor.return_value


def test_close_calls_session_close(token_env):
    """``close()`` releases the underlying Session connection pool."""
    with mock.patch('requests.Session') as mock_session_ctor:
        mock_session_ctor.return_value = mock.MagicMock(headers={})
        cb = _make_cb()
    cb.close()
    cb.session.close.assert_called_once()
    # Idempotent — second call must not raise.
    cb.close()
    # Two close() calls -> two underlying session.close() calls (no internal
    # guard). The contract is "safe to call twice", not "called once
    # internally" — both behaviours are acceptable.
    assert cb.session.close.call_count >= 1


def test_v2_url_format_log(token_env):
    """``on_log`` POSTs to /api/v2/anuga/runs/<id>/log/."""
    cb = _make_cb()
    with mock.patch('run_anuga._http.post_to_control_server') as mock_post:
        cb.on_status('building mesh')
    assert mock_post.call_count == 1
    call_args = mock_post.call_args
    assert call_args.args[0] == 'https://hydrata.com/api/v2/anuga/runs/99/log/'
    # Session is the owned one, not None.
    assert call_args.kwargs.get('session') is cb.session
    # auth kwarg MUST NOT be passed (header on session does the work).
    assert 'auth' not in call_args.kwargs or call_args.kwargs.get('auth') is None
    cb.close()


def test_v2_url_format_progress(token_env):
    """``on_progress`` POSTs to /api/v2/anuga/runs/<id>/progress/."""
    cb = _make_cb()
    with mock.patch('run_anuga._http.post_to_control_server') as mock_post:
        cb.on_progress(42.5, eta_seconds=300)
    assert mock_post.call_count == 1
    call_args = mock_post.call_args
    assert call_args.args[0] == 'https://hydrata.com/api/v2/anuga/runs/99/progress/'
    # Body uses the V2 schema (progress_pct / eta_seconds) per
    # api_v2.py:1102-1116.
    body = call_args.kwargs.get('data') or {}
    assert body['progress_pct'] == 42.5
    assert body['eta_seconds'] == 300
    cb.close()


def test_x_internal_token_header_present(token_env):
    """Pre-set ``X-Internal-Token`` is RAW (no ``Bearer`` prefix)."""
    cb = _make_cb()
    assert cb.session.headers.get('X-Internal-Token') == 'test-token-123'
    # Explicitly NOT prefixed.
    assert 'Bearer ' not in cb.session.headers.get('X-Internal-Token', '')
    cb.close()


def test_no_authorization_header(token_env):
    """No ``Authorization`` header is set — auth flows via X-Internal-Token."""
    cb = _make_cb()
    assert 'Authorization' not in cb.session.headers
    cb.close()


def test_init_raises_on_missing_token(monkeypatch):
    """Fail-fast (AC6): empty or unset token raises RuntimeError in __init__."""
    monkeypatch.delenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', raising=False)
    with pytest.raises(RuntimeError, match='HYDRATA_INTERNAL_COMPUTE_TOKEN'):
        HydrataCallback(
            control_server='https://hydrata.com/',
            project=1,
            scenario=2,
            run_id=3,
        )
    # Empty-string token also fails.
    monkeypatch.setenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', '')
    with pytest.raises(RuntimeError, match='HYDRATA_INTERNAL_COMPUTE_TOKEN'):
        HydrataCallback(
            control_server='https://hydrata.com/',
            project=1,
            scenario=2,
            run_id=3,
        )
