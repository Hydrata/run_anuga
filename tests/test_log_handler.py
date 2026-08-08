"""Log-shipping tests for ``setup_logger`` + the run_anuga logger stack.

TASK-2681 (epic 2662 W4.1) deleted ``_V2LogHandler``, which POSTed records to
``/api/v2/anuga/runs/<id>/log/`` — that route is a 410 tombstone. The ONE web
log channel is now ``_TelemetryLogHandler``, shipping typed ``log`` events
(its behavioural coverage lives in ``test_events_dialect.py``
``TestSetupLoggerEventsAndRankGating``). What this module keeps:

* the handler-lifecycle invariants ``setup_logger`` owes regardless of dialect
  (no duplicate web handlers on re-entry; prior handler closed, not leaked;
  no web handler at all when nothing armed one), and
* the TASK-1276 mname/lnum filter tests, which are about run_anuga's loggers
  formatting cleanly under anuga_core's root formatter and have nothing to do
  with the web channel.
"""

from __future__ import annotations

import logging
from unittest import mock

import requests  # noqa: F401 — asserts requests is importable in the test env

from run_anuga.run_utils import _TelemetryLogHandler, setup_logger


class _FakeTelemetryClient:
    """Duck-typed TelemetryClient — setup_logger only needs .log()."""

    def __init__(self):
        self.lines = []

    def log(self, line):
        self.lines.append(line)
        return True


def _input_data(tmp_path):
    return {
        'output_directory': str(tmp_path),
        'scenario_config': {
            'control_server': 'https://hydrata.com/',
            'run_id': 99,
            'project': 1,
            'id': 2,
        },
    }


def _web_handlers(lg):
    return [h for h in lg.handlers if isinstance(h, _TelemetryLogHandler)]


def _drain(lg):
    for h in lg.handlers[:]:
        lg.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass


def test_setup_logger_installs_the_telemetry_handler_when_a_client_is_armed(
    tmp_path, monkeypatch,
):
    monkeypatch.delenv('OMPI_COMM_WORLD_RANK', raising=False)
    client = _FakeTelemetryClient()
    lg = setup_logger(_input_data(tmp_path), batch_number=1,
                      telemetry_client=client)
    try:
        assert len(_web_handlers(lg)) == 1
    finally:
        _drain(lg)


def test_setup_logger_installs_no_web_handler_without_a_client(
    tmp_path, monkeypatch,
):
    """TASK-2681: a token alone no longer arms a web channel.

    Pre-W4.1 a bare HYDRATA_INTERNAL_COMPUTE_TOKEN installed the legacy
    ``_V2LogHandler``. With that route deleted, the honest degrade for a
    standalone CLI run is file/console logging only — NOT a second dialect.
    """
    monkeypatch.delenv('OMPI_COMM_WORLD_RANK', raising=False)
    monkeypatch.setenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', 'test-token-123')
    lg = setup_logger(_input_data(tmp_path), batch_number=1)
    try:
        assert _web_handlers(lg) == []
    finally:
        _drain(lg)


def test_setup_logger_reentry_closes_prior_handler(tmp_path, monkeypatch):
    """Re-entry removes+closes the prior web handler — exactly one survives.

    run_sim() can be called repeatedly in a long-lived localhost celery worker,
    and a leaked handler means duplicated log events forever.
    """
    monkeypatch.delenv('OMPI_COMM_WORLD_RANK', raising=False)
    lg = setup_logger(_input_data(tmp_path), batch_number=1,
                      telemetry_client=_FakeTelemetryClient())
    first = _web_handlers(lg)[0]
    with mock.patch.object(
        _TelemetryLogHandler, 'close', autospec=True,
    ) as mock_close:
        lg2 = setup_logger(_input_data(tmp_path), batch_number=2,
                           telemetry_client=_FakeTelemetryClient())
        assert any(
            c.args and c.args[0] is first for c in mock_close.call_args_list
        ), 'the prior handler must be closed, not merely dropped'
    try:
        assert len(_web_handlers(lg2)) == 1
    finally:
        _drain(lg2)


def test_setup_logger_emit_swallows_transport_errors(tmp_path, monkeypatch):
    """A telemetry outage must never break the run loop."""
    monkeypatch.delenv('OMPI_COMM_WORLD_RANK', raising=False)

    class _Exploding(_FakeTelemetryClient):
        def log(self, line):
            raise RuntimeError('control server down')

    lg = setup_logger(_input_data(tmp_path), batch_number=1,
                      telemetry_client=_Exploding())
    try:
        lg.info('this must not raise')
    finally:
        _drain(lg)


# --- TASK-1276: run_anuga logs format cleanly under anuga_core's root formatter ---
# anuga_core's basicConfig installs a root formatter referencing %(mname)s /
# %(lnum)s (anuga/utilities/log.py). run_anuga records carry no such fields, so
# when they propagate to that root handler it raises 'KeyError: mname' on every
# emit (the CloudWatch '--- Logging error ---' spam seen in TASK-1182 W2 canary
# 19). The fix keeps propagation ON (so pytest caplog still captures run_anuga
# records) and stamps mname/lnum via a filter (run_anuga/_logging.py).

# The exact (old-style) format string anuga_core installs on the root logger.
_ANUGA_ROOT_FMT = '%(asctime)s %(levelname)-8s %(mname)25s:%(lnum)-4d|%(message)s'


class _MnameRootHandler(logging.Handler):
    """Mimics anuga_core's root handler: formats with %(mname)s/%(lnum)s.

    Records every LogRecord it successfully formats and any formatting error, so
    a test can assert run_anuga records both reach it (propagation intact) and
    render without KeyError: mname.
    """

    def __init__(self):
        super().__init__()
        self.received = []
        self.format_errors = []
        self.setFormatter(logging.Formatter(_ANUGA_ROOT_FMT))

    def emit(self, record):
        try:
            self.format(record)
            self.received.append(record)
        except Exception as exc:  # the KeyError: 'mname' this fix prevents
            self.format_errors.append(exc)


def test_mname_filter_installed_on_all_run_anuga_loggers():
    """Every run_anuga logger that emits carries the mname/lnum filter."""
    from run_anuga._logging import MnameLnumFilter
    from run_anuga import run, run_utils, callbacks, _http
    for module in (run, run_utils, callbacks, _http):
        assert any(isinstance(f, MnameLnumFilter) for f in module.logger.filters), \
            f"{module.__name__}.logger is missing MnameLnumFilter"


def test_run_anuga_emit_formats_under_anuga_root_formatter():
    """A run_anuga emit reaches anuga_core's %(mname)s root formatter (propagation
    intact for caplog) and renders without KeyError: mname (TASK-1276)."""
    from run_anuga import run_utils as run_utils_module
    lg = run_utils_module.logger
    root = logging.getLogger()
    recorder = _MnameRootHandler()
    # Isolate: clear lg's own handlers so a leaked file/V2 handler can't fire (no
    # disk/network). Filters live on lg.filters, so the mname filter still runs.
    saved_handlers = lg.handlers[:]
    lg.handlers = []
    root.addHandler(recorder)
    try:
        lg.error('evolving timestep 1/100')
    finally:
        root.removeHandler(recorder)
        lg.handlers = saved_handlers
    # No KeyError: mname when anuga's root formatter renders the record.
    assert recorder.format_errors == []
    # Propagation preserved: the record actually reached the root handler.
    assert any('evolving timestep' in r.getMessage() for r in recorder.received)
