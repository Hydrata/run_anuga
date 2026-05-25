"""Tests for the V2 log handler installed by ``setup_logger`` (TASK-989).

Before TASK-989, ``setup_logger`` installed a stdlib
``logging.handlers.HTTPHandler`` that POSTed log lines via V1 BasicAuth to
``/anuga/api/{p}/{s}/run/{r}/log/``. That channel only fired on localhost (the
Batch entrypoint passes no creds) and 401'd against allauth (rid=24302's 13x
401 storm). It is now replaced by ``_V2LogHandler``, which mirrors
``HydrataCallback``: a single owned ``requests.Session`` carrying the raw
``X-Internal-Token`` header, POSTing ``{message, levelname, created}`` to
``/api/v2/anuga/runs/<id>/log/`` via ``_http.post_to_control_server``.

Plain ``import requests`` + ``from run_anuga... import`` at module top (NO
importorskip), matching test_http_helper.py / test_callbacks_v2.py.
"""

from __future__ import annotations

import logging
from unittest import mock

import requests  # noqa: F401 — asserts requests is importable in the test env

from run_anuga.run_utils import _V2LogHandler, setup_logger


def _make_handler():
    return _V2LogHandler(
        control_server='https://hydrata.com/',
        run_id=99,
        token='test-token-123',
    )


def _make_record(msg='hello', level=logging.INFO):
    return logging.LogRecord(
        name='run_anuga.test',
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=None,
        exc_info=None,
    )


def test_init_creates_session_once():
    """One ``requests.Session()`` per handler instance (owned)."""
    with mock.patch('requests.Session') as mock_session_ctor:
        mock_session_ctor.return_value = mock.MagicMock(headers={})
        h = _make_handler()
    assert mock_session_ctor.call_count == 1
    assert h._session is mock_session_ctor.return_value


def test_x_internal_token_header_raw_no_bearer():
    """Pre-set ``X-Internal-Token`` is RAW (no ``Bearer`` prefix)."""
    h = _make_handler()
    assert h._session.headers.get('X-Internal-Token') == 'test-token-123'
    assert 'Bearer ' not in h._session.headers.get('X-Internal-Token', '')
    h.close()


def test_no_authorization_header():
    """No ``Authorization`` / BasicAuth header — auth flows via X-Internal-Token."""
    h = _make_handler()
    assert 'Authorization' not in h._session.headers
    # session.auth is left unset (post_to_control_server is called with no auth kwarg)
    assert h._session.auth is None
    h.close()


def test_v2_log_url_shape():
    """The handler targets /api/v2/anuga/runs/<id>/log/ — not the V1 URL."""
    h = _make_handler()
    assert h._log_url == 'https://hydrata.com/api/v2/anuga/runs/99/log/'
    # Explicitly NOT the deleted V1 template.
    assert '/anuga/api/' not in h._log_url
    h.close()


def test_emit_posts_v2_payload_via_owned_session():
    """emit() delegates to post_to_control_server with the V2 body + owned session, no auth kwarg."""
    h = _make_handler()
    with mock.patch('run_anuga._http.post_to_control_server') as mock_post:
        h.emit(_make_record(msg='evolving timestep', level=logging.WARNING))
    assert mock_post.call_count == 1
    call_args = mock_post.call_args
    # URL is the V2 log endpoint.
    assert call_args.args[0] == 'https://hydrata.com/api/v2/anuga/runs/99/log/'
    # Owned session, POST method, no BasicAuth.
    assert call_args.kwargs.get('session') is h._session
    assert call_args.kwargs.get('method') == 'POST'
    assert 'auth' not in call_args.kwargs or call_args.kwargs.get('auth') is None
    # V2 body shape matches HydrataCallback / api_v2 /log/ contract.
    body = call_args.kwargs.get('data') or {}
    assert body['message'] == 'evolving timestep'
    assert body['levelname'] == 'WARNING'
    assert 'created' in body and isinstance(body['created'], float)
    h.close()


def test_emit_swallows_transport_errors():
    """A transport failure must never propagate out of emit() (run loop safety)."""
    h = _make_handler()
    with mock.patch('run_anuga._http.post_to_control_server', side_effect=requests.ConnectionError('boom')):
        with mock.patch.object(h, 'handleError') as mock_handle:
            h.emit(_make_record())  # must not raise
    mock_handle.assert_called_once()
    h.close()


def test_close_releases_session_and_is_idempotent():
    """close() releases the owned Session connection pool; safe to call twice."""
    with mock.patch('requests.Session') as mock_session_ctor:
        mock_session_ctor.return_value = mock.MagicMock(headers={})
        h = _make_handler()
    h.close()
    h._session.close.assert_called_once()
    # Idempotent — second call must not raise.
    h.close()
    assert h._session.close.call_count >= 1


# --- setup_logger integration: V2 handler only, no V1 BasicAuth HTTPHandler ---


def _input_data(tmp_path):
    return {
        'output_directory': str(tmp_path),
        'scenario_config': {
            'control_server': 'https://hydrata.com/',
            'project': 42,
            'id': 7,
            'run_id': 99,
        },
    }


def _http_handlers(lg):
    return [h for h in lg.handlers if isinstance(h, (logging.handlers.HTTPHandler, _V2LogHandler))]


def test_setup_logger_installs_v2_handler_when_token_present(tmp_path, monkeypatch):
    """With the token set, setup_logger installs a _V2LogHandler (not a V1 HTTPHandler)."""
    monkeypatch.setenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', 'test-token-123')
    lg = setup_logger(_input_data(tmp_path), batch_number=1)
    try:
        net = _http_handlers(lg)
        assert len(net) == 1
        handler = net[0]
        assert isinstance(handler, _V2LogHandler)
        # No legacy V1 BasicAuth HTTPHandler installed.
        assert not isinstance(handler, logging.handlers.HTTPHandler)
        assert handler._log_url == 'https://hydrata.com/api/v2/anuga/runs/99/log/'
        assert handler._session.headers.get('X-Internal-Token') == 'test-token-123'
    finally:
        for h in lg.handlers[:]:
            lg.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass


def test_setup_logger_no_web_handler_without_token(tmp_path, monkeypatch):
    """No token (standalone CLI run) -> no network handler installed; file-only."""
    monkeypatch.delenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', raising=False)
    monkeypatch.delenv('COMPUTE_USERNAME', raising=False)
    monkeypatch.delenv('COMPUTE_PASSWORD', raising=False)
    lg = setup_logger(_input_data(tmp_path), batch_number=1)
    try:
        assert _http_handlers(lg) == []
    finally:
        for h in lg.handlers[:]:
            lg.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass


def test_setup_logger_never_uses_basicauth(tmp_path, monkeypatch):
    """Even with legacy COMPUTE_USERNAME/PASSWORD set, no BasicAuth V1 handler appears."""
    monkeypatch.setenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', 'test-token-123')
    monkeypatch.setenv('COMPUTE_USERNAME', 'legacy-user')
    monkeypatch.setenv('COMPUTE_PASSWORD', 'legacy-pass')
    lg = setup_logger(_input_data(tmp_path), username='legacy-user', password='legacy-pass', batch_number=1)
    try:
        net = _http_handlers(lg)
        assert len(net) == 1
        # The handler is the V2 token handler, and carries no BasicAuth.
        assert isinstance(net[0], _V2LogHandler)
        assert 'Authorization' not in net[0]._session.headers
    finally:
        for h in lg.handlers[:]:
            lg.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass


def test_setup_logger_reentry_closes_prior_handler(tmp_path, monkeypatch):
    """Calling setup_logger twice removes+closes the prior _V2LogHandler (no leak, one survives)."""
    monkeypatch.setenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', 'test-token-123')
    lg = setup_logger(_input_data(tmp_path), batch_number=1)
    first = _http_handlers(lg)[0]
    with mock.patch.object(_V2LogHandler, 'close', autospec=True) as mock_close:
        lg2 = setup_logger(_input_data(tmp_path), batch_number=2)
        # The first handler's close() was invoked during re-entry cleanup.
        assert any(c.args and c.args[0] is first for c in mock_close.call_args_list)
    # Exactly one network handler survives after re-entry.
    try:
        assert len(_http_handlers(lg2)) == 1
    finally:
        for h in lg2.handlers[:]:
            lg2.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
