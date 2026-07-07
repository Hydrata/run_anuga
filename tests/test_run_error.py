"""Tests for run_anuga.run._report_run_error — TASK-1078 (W6.2 of TASK-1048).

`_report_run_error` POSTs the originating run failure to
``/api/v2/anuga/runs/<id>/error/`` and is invoked from the top-level
``main()`` exception handler. Without these tests, canary 15 wedged silently
because the function had no Batch token-auth path: HYDRATA_INTERNAL_COMPUTE_TOKEN
was set in the container but the function only knew about HTTPBasicAuth, and
the BE rejected un-authed POSTs with 401, swallowing the error report.

Coverage:
* Token-auth path (HYDRATA_INTERNAL_COMPUTE_TOKEN set) — POSTs with
  ``X-Internal-Token`` header on a session, no ``auth=`` kwarg.
* Legacy BasicAuth path (username/password, no token) — POSTs with
  ``auth=HTTPBasicAuth(...)``, no ``session=`` kwarg.
* Token takes precedence over BasicAuth — token-mode wins even if creds present.
* No-creds-and-no-token early returns without POSTing (defensive — caller
  passes user-supplied CLI args; the function must not 401-loop).
* Outer try/except swallows post failures (must never mask the originating
  run exception).
"""

from __future__ import annotations

import json
from unittest import mock

import pytest


@pytest.fixture
def fake_package(tmp_path):
    """A package_dir with a minimal scenario.json (run_id + control_server)."""
    package = tmp_path / 'pkg'
    package.mkdir()
    (package / 'scenario.json').write_text(json.dumps({
        'run_id': 42,
        'control_server': 'https://example.test/',
    }))
    return str(package)


def test_run_error_token_path(fake_package, monkeypatch):
    """Token set → POST with X-Internal-Token header (RAW), no auth= kwarg."""
    monkeypatch.setenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', 'tok123')
    from run_anuga.run import _report_run_error
    from run_anuga.run_utils import RunContext

    with mock.patch('run_anuga._http.post_to_control_server') as mocked:
        _report_run_error(RunContext(fake_package, None, None), 'boom')

    assert mocked.call_count == 1
    args, kwargs = mocked.call_args
    assert args[0] == 'https://example.test/api/v2/anuga/runs/42/error/'
    assert kwargs.get('method') == 'POST'
    assert kwargs.get('data') == {'message': 'boom'}
    assert kwargs.get('timeout') == 30
    # Token-mode contract: session passed in with pre-set header, NO auth= kwarg.
    session = kwargs.get('session')
    assert session is not None
    assert session.headers.get('X-Internal-Token') == 'tok123'
    # Explicit RAW token — never "Bearer <token>".
    assert not session.headers.get('X-Internal-Token', '').startswith('Bearer ')
    assert kwargs.get('auth') is None


def test_run_error_username_password_path(fake_package, monkeypatch):
    """Token unset + username/password set → BasicAuth path still works."""
    monkeypatch.delenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', raising=False)
    from run_anuga.run import _report_run_error
    from run_anuga.run_utils import RunContext

    with mock.patch('run_anuga._http.post_to_control_server') as mocked:
        _report_run_error(RunContext(fake_package, 'user@test', 'pw'), 'boom')

    assert mocked.call_count == 1
    args, kwargs = mocked.call_args
    assert args[0] == 'https://example.test/api/v2/anuga/runs/42/error/'
    assert kwargs.get('method') == 'POST'
    assert kwargs.get('data') == {'message': 'boom'}
    assert kwargs.get('timeout') == 30
    # BasicAuth-mode contract: auth= kwarg set, session= not set.
    auth = kwargs.get('auth')
    assert auth is not None
    assert getattr(auth, 'username', None) == 'user@test'
    assert getattr(auth, 'password', None) == 'pw'
    assert kwargs.get('session') is None


def test_run_error_token_takes_precedence_over_basic_auth(fake_package, monkeypatch):
    """Token + creds both present → token wins (BasicAuth not attempted)."""
    monkeypatch.setenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', 'tok-wins')
    from run_anuga.run import _report_run_error
    from run_anuga.run_utils import RunContext

    with mock.patch('run_anuga._http.post_to_control_server') as mocked:
        _report_run_error(RunContext(fake_package, 'user@test', 'pw'), 'boom')

    assert mocked.call_count == 1
    _, kwargs = mocked.call_args
    session = kwargs.get('session')
    assert session is not None
    assert session.headers.get('X-Internal-Token') == 'tok-wins'
    # Confirm BasicAuth is NOT attempted (no auth= kwarg in token mode).
    assert kwargs.get('auth') is None


def test_run_error_no_creds_no_token_returns(fake_package, monkeypatch):
    """No token AND no creds → silent early return (no POST attempted)."""
    monkeypatch.delenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', raising=False)
    from run_anuga.run import _report_run_error
    from run_anuga.run_utils import RunContext

    with mock.patch('run_anuga._http.post_to_control_server') as mocked:
        _report_run_error(RunContext(fake_package, None, None), 'boom')

    assert mocked.call_count == 0


def test_run_error_swallows_exceptions(fake_package, monkeypatch, caplog):
    """Outer try/except must never propagate — would mask the run exception."""
    import logging
    monkeypatch.setenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', 'tok-raise')
    from run_anuga.run import _report_run_error
    from run_anuga.run_utils import RunContext

    with mock.patch(
        'run_anuga._http.post_to_control_server',
        side_effect=RuntimeError('network kaboom'),
    ):
        with caplog.at_level(logging.ERROR, logger='run_anuga.run'):
            # Must NOT raise.
            _report_run_error(RunContext(fake_package, None, None), 'boom')

    # The failure was logged at ERROR level via logger.exception.
    assert 'Failed to report run error to control server' in caplog.text
