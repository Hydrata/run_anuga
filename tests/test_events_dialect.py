"""TASK-2672 (epic 2662 W2.2) — run_anuga speaks events.

Covers the events-dialect wiring end to end (with a stubbed
``gn_anuga.batch_common`` — the real client lib is baked into the images and
is deliberately NOT importable from a bare run_anuga env):

* ``TelemetryCallback`` maps the SimulationCallback protocol onto typed events.
* ``_make_telemetry_client`` — the ONE explicit construction site. TASK-2692
  (W5) made it FAIL-CLOSED: absent ``HYDRATA_PROCESS_ID`` / unimportable
  batch_common now RAISE (there is no legacy dialect left to degrade to), with
  one explicit env opt-out for deliberately-unreported ad-hoc runs.
* ``run_and_report`` arms started(placement)/watchdog/phase-listener, posts
  the result event (no legacy fallback since W5 — an undeliverable terminal
  event raises), posts the error event on a sim crash AND on a zip/upload
  failure, and tears the channels down in ``finally``.
* ``phase_tracker.set_phase_listener`` — transition events, None skipped,
  listener exceptions swallowed, ``reset()`` keeps the registration.
* ``setup_logger`` — web/telemetry log shipping is rank-0 only (the 32x
  startup-spam kill), and ships via ``_TelemetryLogHandler`` when the events
  client is armed.
* D7: the container computes NO ETA (source pin + behavioural check).
"""
from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

from run_anuga import phase_tracker
from run_anuga._handoff import (
    ALLOW_UNREPORTED_ENV,
    _make_telemetry_client,
    run_and_report,
)
from run_anuga.callbacks import TelemetryCallback


class FakeTelemetryClient:
    """Records every typed-event call; posts 'succeed' by default."""

    def __init__(self, control_server=None, process_id=None, token=None, **kwargs):
        if not control_server:
            raise ValueError('TelemetryClient: control_server is required')
        if not process_id:
            raise ValueError('TelemetryClient: process_id is required')
        if not token:
            raise ValueError('TelemetryClient: token is required')
        self.control_server = control_server
        self.process_id = process_id
        self.token = token
        self.calls = []
        self.result_ok = True
        self.error_ok = True
        self.watchdog_running = False

    def started(self, hardware=None, **fields):
        self.calls.append(('started', hardware))
        return True

    def progress(self, pct, eta_seconds=None, detail=None):
        self.calls.append(('progress', pct, eta_seconds))
        return True

    def phase(self, phase):
        self.calls.append(('phase', phase))
        return True

    def log(self, line):
        self.calls.append(('log', line))
        return True

    def metric(self, metrics):
        self.calls.append(('metric', metrics))
        return True

    def result(self, **fields):
        self.calls.append(('result', fields))
        return self.result_ok

    def error(self, message):
        self.calls.append(('error', message))
        return self.error_ok

    def start_watchdog(self, probe_provider=None, interval_s=60):
        self.calls.append(('start_watchdog', probe_provider))
        self.watchdog_running = True

    def stop_watchdog(self, join_timeout_s=5):
        self.calls.append(('stop_watchdog',))
        self.watchdog_running = False

    # helpers -------------------------------------------------------------
    def names(self):
        return [c[0] for c in self.calls]

    def of(self, name):
        return [c for c in self.calls if c[0] == name]


def _install_stub_batch_common(monkeypatch, hardware=None):
    """Register stub gn_anuga.batch_common modules in sys.modules."""
    pkg = types.ModuleType('gn_anuga')
    sub = types.ModuleType('gn_anuga.batch_common')
    client_mod = types.ModuleType('gn_anuga.batch_common.telemetry_client')
    client_mod.TelemetryClient = FakeTelemetryClient
    hw_mod = types.ModuleType('gn_anuga.batch_common.hardware_identity')
    hw_mod.collect_hardware_identity = lambda **kw: (
        hardware if hardware is not None else {'instance_type': 'stub.large'}
    )
    monkeypatch.setitem(sys.modules, 'gn_anuga', pkg)
    monkeypatch.setitem(sys.modules, 'gn_anuga.batch_common', sub)
    monkeypatch.setitem(
        sys.modules, 'gn_anuga.batch_common.telemetry_client', client_mod)
    monkeypatch.setitem(
        sys.modules, 'gn_anuga.batch_common.hardware_identity', hw_mod)
    # resource_sampler stays ABSENT on purpose: _make_resource_sampler's
    # guarded import must keep degrading to None (localhost shape).
    return client_mod


@pytest.fixture(autouse=True)
def _clean_listener():
    yield
    phase_tracker.set_phase_listener(None)
    phase_tracker.reset()


# ---------------------------------------------------------------------------
# TelemetryCallback mapping
# ---------------------------------------------------------------------------

class TestTelemetryCallbackMapping:
    def _client(self):
        return FakeTelemetryClient('http://cs', 'proc-1', 'tok')

    def test_progress_maps_to_progress_event(self):
        client = self._client()
        TelemetryCallback(client).on_progress(42.5)
        assert client.calls == [('progress', 42.5, None)]

    def test_progress_passes_container_eta_override_through(self):
        client = self._client()
        TelemetryCallback(client).on_progress(10.0, eta_seconds=120)
        assert client.calls == [('progress', 10.0, 120)]

    def test_status_and_file_ride_the_log_channel(self):
        client = self._client()
        cb = TelemetryCallback(client)
        cb.on_status('building mesh')
        cb.on_file('video', '/tmp/x.mp4')
        assert client.calls == [
            ('log', 'status: building mesh'),
            ('log', 'file: video -> /tmp/x.mp4'),
        ]

    def test_metric_maps_to_metric_event(self):
        client = self._client()
        TelemetryCallback(client).on_metric('memory_used', 123)
        assert client.calls == [('metric', {'memory_used': 123})]

    def test_close_and_mesh_features_are_noops(self):
        client = self._client()
        cb = TelemetryCallback(client)
        cb.on_mesh_features_ready()
        cb.close()
        assert client.calls == []


# ---------------------------------------------------------------------------
# _make_telemetry_client — the one construction site
# ---------------------------------------------------------------------------

class TestMakeTelemetryClient:
    """TASK-2692 (W5): the construction site is FAIL-CLOSED.

    Both former ``return None`` degrades handed the run back to the legacy
    ``/process-result/`` + ``/error/`` dialect. W5 deletes that dialect and
    tombstones the routes, so ``None`` now means NO channel at all — the
    container would compute for hours and bin the result. Both conditions
    raise, and the only escape is the explicit opt-out env var.
    """

    def test_no_process_id_raises(self, monkeypatch):
        monkeypatch.delenv('HYDRATA_PROCESS_ID', raising=False)
        monkeypatch.delenv(ALLOW_UNREPORTED_ENV, raising=False)
        with pytest.raises(RuntimeError, match='HYDRATA_PROCESS_ID is not set'):
            _make_telemetry_client({'control_server': 'http://cs'})

    def test_no_process_id_message_names_both_dispatchers(self, monkeypatch):
        """The operator must be told WHERE to look — both dispatch paths."""
        monkeypatch.delenv('HYDRATA_PROCESS_ID', raising=False)
        monkeypatch.delenv(ALLOW_UNREPORTED_ENV, raising=False)
        with pytest.raises(RuntimeError) as excinfo:
            _make_telemetry_client({'control_server': 'http://cs'})
        message = str(excinfo.value)
        assert '_dispatch_batch' in message
        assert 'dispatch_local_anuga_run' in message
        assert ALLOW_UNREPORTED_ENV in message

    def test_unimportable_batch_common_raises(self, monkeypatch):
        monkeypatch.setenv('HYDRATA_PROCESS_ID', 'proc-uuid')
        monkeypatch.delenv(ALLOW_UNREPORTED_ENV, raising=False)
        for name in list(sys.modules):
            if name == 'gn_anuga' or name.startswith('gn_anuga.'):
                monkeypatch.delitem(sys.modules, name, raising=False)
        with pytest.raises(RuntimeError, match='not importable'):
            _make_telemetry_client({'control_server': 'http://cs'})

    @pytest.mark.parametrize('value', ['1', 'true', 'TRUE', 'yes', 'on'])
    def test_opt_out_downgrades_missing_process_id_to_a_loud_error(
            self, monkeypatch, caplog, value):
        """(Was ..._to_a_warning.) The downgrade must log at ERROR, not WARNING.

        W5 adversarial review: this opt-out is INHERITABLE — the celery-native
        dispatcher builds the child env as ``dict(os.environ)``, so an operator
        who exports the variable in the shell that restarts celery arms it for
        every subsequent run in that worker. Such a run reports NOTHING, so
        there is no server-side signal at all and this log line is the only
        evidence a later "why did this run go dark" investigation can find. It
        must therefore be ERROR-level and must name the variable.
        """
        monkeypatch.delenv('HYDRATA_PROCESS_ID', raising=False)
        monkeypatch.setenv(ALLOW_UNREPORTED_ENV, value)
        with caplog.at_level(logging.WARNING, logger='run_anuga._handoff'):
            client = _make_telemetry_client({'control_server': 'http://cs'})
        assert client is None
        assert 'NO telemetry channel' in caplog.text

        records = [r for r in caplog.records if r.name == 'run_anuga._handoff']
        assert records, 'nothing was logged at all'
        assert [r.levelno for r in records] == [logging.ERROR], (
            'the unreported-run downgrade must be ERROR, not WARNING: %r'
            % ([logging.getLevelName(r.levelno) for r in records],)
        )
        text = records[0].getMessage()
        assert ALLOW_UNREPORTED_ENV in text, text
        assert 'Nothing (result OR error) will be reported' in text, text

    @pytest.mark.parametrize('value', ['', '0', 'false', 'no', 'off', 'maybe'])
    def test_non_truthy_opt_out_values_stay_fail_closed(self, monkeypatch, value):
        """A stray `RUN_ANUGA_ALLOW_UNREPORTED_RUN=0` in a job definition must
        not accidentally disarm the lock."""
        monkeypatch.delenv('HYDRATA_PROCESS_ID', raising=False)
        monkeypatch.setenv(ALLOW_UNREPORTED_ENV, value)
        with pytest.raises(RuntimeError, match='HYDRATA_PROCESS_ID is not set'):
            _make_telemetry_client({'control_server': 'http://cs'})

    def test_opt_out_downgrades_unimportable_batch_common(
            self, monkeypatch, caplog):
        monkeypatch.setenv('HYDRATA_PROCESS_ID', 'proc-uuid')
        monkeypatch.setenv(ALLOW_UNREPORTED_ENV, '1')
        for name in list(sys.modules):
            if name == 'gn_anuga' or name.startswith('gn_anuga.'):
                monkeypatch.delitem(sys.modules, name, raising=False)
        with caplog.at_level(logging.WARNING, logger='run_anuga._handoff'):
            client = _make_telemetry_client({'control_server': 'http://cs'})
        assert client is None
        assert 'not importable' in caplog.text

        # Same reasoning as the missing-process-id twin above: ERROR, naming
        # the variable, because nothing reaches the server on this path either.
        records = [r for r in caplog.records if r.name == 'run_anuga._handoff']
        assert [r.levelno for r in records] == [logging.ERROR], (
            [logging.getLevelName(r.levelno) for r in records]
        )
        assert ALLOW_UNREPORTED_ENV in records[0].getMessage()

    def test_cli_run_never_reaches_the_construction_site(
            self, tmp_path, monkeypatch):
        """The ad-hoc escape is STRUCTURAL, not a flag.

        ``run_anuga.cli run`` calls ``run_sim`` directly and never builds a
        telemetry client, so the fail-closed raise cannot reach it even with
        NO env vars set at all. This is why the gate is "run_and_report is
        fail-closed" and not an ``AWS_BATCH_JOB_ID`` context sniff — the
        celery-native localhost dispatcher shells ``run-and-report`` WITHOUT
        that variable, so a Batch-context gate would have left the entire
        localhost path silently unarmed.
        """
        from run_anuga import cli

        monkeypatch.delenv('HYDRATA_PROCESS_ID', raising=False)
        monkeypatch.delenv(ALLOW_UNREPORTED_ENV, raising=False)
        args = types.SimpleNamespace(
            package_dir=str(tmp_path), username=None, password=None,
            batch_number=None, checkpoint_time=None, log_to_stdout=False,
        )
        with mock.patch('run_anuga.run.run_sim') as mock_run_sim:
            cli.cmd_run(args)  # must NOT raise
        mock_run_sim.assert_called_once()
        assert 'telemetry_client' not in mock_run_sim.call_args.kwargs

    def test_constructs_client_with_config(self, monkeypatch):
        monkeypatch.setenv('HYDRATA_PROCESS_ID', 'proc-uuid')
        monkeypatch.setenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', 'tok')
        _install_stub_batch_common(monkeypatch)
        client = _make_telemetry_client({'control_server': 'http://cs/'})
        assert isinstance(client, FakeTelemetryClient)
        assert client.process_id == 'proc-uuid'
        assert client.control_server == 'http://cs/'
        assert client.token == 'tok'

    def test_missing_token_with_process_id_fails_loud(self, monkeypatch):
        """A misconfigured container dies at startup, never runs dark."""
        monkeypatch.setenv('HYDRATA_PROCESS_ID', 'proc-uuid')
        monkeypatch.delenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', raising=False)
        _install_stub_batch_common(monkeypatch)
        with pytest.raises(ValueError, match='token'):
            _make_telemetry_client({'control_server': 'http://cs/'})


# ---------------------------------------------------------------------------
# phase_tracker listener
# ---------------------------------------------------------------------------

class TestPhaseListener:
    def test_listener_fires_on_transitions_and_skips_none(self):
        seen = []
        phase_tracker.set_phase_listener(seen.append)
        phase_tracker.set_phase(phase_tracker.PHASE_MESH_GEN)
        phase_tracker.set_phase(None)
        phase_tracker.set_phase(phase_tracker.PHASE_EVOLVE)
        phase_tracker.set_phase(phase_tracker.PHASE_EVOLVE)  # no change
        assert seen == [phase_tracker.PHASE_MESH_GEN, phase_tracker.PHASE_EVOLVE]

    def test_listener_exception_never_breaks_set_phase(self):
        def boom(_):
            raise RuntimeError('listener crash')
        phase_tracker.set_phase_listener(boom)
        phase_tracker.set_phase(phase_tracker.PHASE_MESH_GEN)  # must not raise
        assert phase_tracker.get_phase() == phase_tracker.PHASE_MESH_GEN

    def test_reset_keeps_the_registration(self):
        """run_and_report registers BEFORE run_sim, which calls reset() —
        reset clears phase state, never the listener."""
        seen = []
        phase_tracker.set_phase_listener(seen.append)
        phase_tracker.reset()
        phase_tracker.set_phase(phase_tracker.PHASE_DISTRIBUTE)
        assert seen == [phase_tracker.PHASE_DISTRIBUTE]


# ---------------------------------------------------------------------------
# run_and_report — events dialect end to end (run_sim mocked)
# ---------------------------------------------------------------------------

@pytest.fixture
def package(tmp_path: Path) -> Path:
    config = {
        'id': 384,
        'project': 601,
        'run_id': 1243,
        'control_server': 'https://hydrata.com/',
    }
    (tmp_path / 'scenario.json').write_text(json.dumps(config))
    outputs = tmp_path / 'outputs_601_384_1243'
    outputs.mkdir()
    (outputs / 'result_depth_max.tif').write_bytes(b'fake tif')
    return tmp_path


@pytest.fixture
def events_env(monkeypatch):
    monkeypatch.setenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', 'test-token')
    monkeypatch.setenv('HYDRATA_PROCESS_ID', 'proc-uuid-2672')
    _install_stub_batch_common(monkeypatch, hardware={'instance_type': 'r7a.8xlarge'})


def _capture_client(mock_run_sim):
    (_, kwargs) = mock_run_sim.call_args
    return kwargs['telemetry_client']


class TestRunAndReportEventsDialect:
    def _patches(self, mock_run_sim):
        # TASK-2692 (W5): report_result/report_error are DELETED — there is no
        # legacy dialect left to patch out.
        from run_anuga import _handoff
        return [
            mock.patch('run_anuga.run.run_sim', mock_run_sim),
            mock.patch.object(_handoff, 'upload_cold_archive'),
            mock.patch.object(_handoff, 'upload_result_to_s3'),
        ]

    def test_events_dialect_full_wiring(self, package, events_env):
        mock_run_sim = mock.MagicMock(return_value=None)
        patches = self._patches(mock_run_sim)
        with patches[0], patches[1], patches[2]:
            out = run_and_report(package, result_bucket='bucket')

        client = _capture_client(mock_run_sim)
        assert isinstance(client, FakeTelemetryClient)
        # the callback wrapper's inner is the events adapter over THIS client
        wrapper = mock_run_sim.call_args.kwargs['callback']
        assert isinstance(wrapper._inner, TelemetryCallback)
        assert wrapper._inner.client is client

        # started event carries the W1.3 placement
        assert client.of('started') == [
            ('started', {'instance_type': 'r7a.8xlarge'})]
        # watchdog armed then stopped (teardown in finally)
        assert 'start_watchdog' in client.names()
        assert client.names()[-1] == 'stop_watchdog'
        assert client.watchdog_running is False
        # the handoff's own archive phase rode the events channel
        assert ('phase', phase_tracker.PHASE_ARCHIVE) in client.calls
        # phase listener unhooked after the run — a new transition adds nothing
        n_phase = len(client.of('phase'))
        phase_tracker.set_phase(phase_tracker.PHASE_MESH_GEN)
        assert len(client.of('phase')) == n_phase

        # ONE result event with key + cold prefix; no error event on success
        assert client.of('result') == [('result', {
            'result_package_key': '601_384_1243_results.zip',
            'cold_archive_prefix': 'cold-archive/601_384_1243/',
        })]
        assert client.of('error') == []
        assert out == {'result_key': '601_384_1243_results.zip',
                       'process_result_status': 'events'}

    def test_phase_transitions_reach_the_client_during_run_sim(
            self, package, events_env):
        def fake_sim(*args, **kwargs):
            phase_tracker.set_phase(phase_tracker.PHASE_MESH_GEN)
            phase_tracker.set_phase(None)

        mock_run_sim = mock.MagicMock(side_effect=fake_sim)
        patches = self._patches(mock_run_sim)
        with patches[0], patches[1], patches[2]:
            run_and_report(package, result_bucket='bucket')
        client = _capture_client(mock_run_sim)
        # mesh-gen (from inside run_sim) first, then the handoff's own archive
        assert client.of('phase')[0] == ('phase', phase_tracker.PHASE_MESH_GEN)

    def test_undeliverable_result_event_raises_naming_the_s3_key(
            self, package, events_env):
        """TASK-2692 (W5): the legacy /process-result/ wedge-defence fallback is
        deleted with its 410'd route. An undeliverable terminal result event now
        RAISES — a non-zero container makes the Batch job FAILED (a signal the
        reaper and an operator can both see) instead of a SUCCEEDED job that
        looks like a wedge for an hour. The zip IS in S3, so the message must
        name the key for hand-recovery.
        """
        mock_run_sim = mock.MagicMock(return_value=None)

        real_make = _make_telemetry_client

        def make_failing(scenario_config):
            client = real_make(scenario_config)
            client.result_ok = False
            return client

        from run_anuga import _handoff
        patches = self._patches(mock_run_sim)
        with mock.patch.object(_handoff, '_make_telemetry_client', make_failing), \
                patches[0], patches[1], patches[2]:
            with pytest.raises(RuntimeError) as excinfo:
                run_and_report(package, result_bucket='bucket')
        message = str(excinfo.value)
        assert 's3://bucket/601_384_1243_results.zip' in message
        client = _capture_client(mock_run_sim)
        # best-effort: still tries to flip the run to ERROR carrying that message
        assert len(client.of('error')) == 1
        assert 's3://bucket/601_384_1243_results.zip' in client.of('error')[0][1]
        # channels still torn down on this path
        assert client.names()[-1] == 'stop_watchdog'

    def test_sim_crash_posts_error_event(self, package, events_env):
        mock_run_sim = mock.MagicMock(side_effect=RuntimeError('sim exploded'))
        patches = self._patches(mock_run_sim)
        with patches[0], patches[1], patches[2]:
            with pytest.raises(RuntimeError, match='sim exploded'):
                run_and_report(package, result_bucket='bucket')
        client = _capture_client(mock_run_sim)
        errors = client.of('error')
        assert len(errors) == 1 and 'sim exploded' in errors[0][1]
        # teardown still ran on the crash path
        assert client.names()[-1] == 'stop_watchdog'

    def test_sim_crash_error_event_failure_still_re_raises(
            self, package, events_env):
        """A dropped terminal error event is NOT masked and has no legacy
        fallback left: the client's bounded retry (spec §2.4) is the
        in-protocol replacement, the W3.3 reaper is the outer net, and the
        originating exception still propagates so the container exits non-zero.
        """
        mock_run_sim = mock.MagicMock(side_effect=RuntimeError('sim exploded'))

        real_make = _make_telemetry_client

        def make_failing(scenario_config):
            client = real_make(scenario_config)
            client.error_ok = False
            return client

        from run_anuga import _handoff
        patches = self._patches(mock_run_sim)
        with mock.patch.object(_handoff, '_make_telemetry_client', make_failing), \
                patches[0], patches[1], patches[2]:
            with pytest.raises(RuntimeError, match='sim exploded'):
                run_and_report(package, result_bucket='bucket')
        client = _capture_client(mock_run_sim)
        assert len(client.of('error')) == 1

    def test_handoff_zip_upload_failure_posts_an_error_event(
            self, package, events_env):
        """TASK-2692: this site POSTed the legacy /error/ route with NO events
        attempt at all before W5 — one of the three call sites that had to be
        CONVERTED, not merely unhooked."""
        from run_anuga import _handoff
        mock_run_sim = mock.MagicMock(return_value=None)
        patches = self._patches(mock_run_sim)
        with patches[0], patches[1], \
                mock.patch.object(_handoff, 'upload_result_to_s3',
                                  side_effect=RuntimeError('s3 exploded')):
            with pytest.raises(RuntimeError, match='s3 exploded'):
                run_and_report(package, result_bucket='bucket')
        client = _capture_client(mock_run_sim)
        errors = client.of('error')
        assert len(errors) == 1, errors
        assert 'run_and_report handoff failed' in errors[0][1]
        assert 's3 exploded' in errors[0][1]
        assert client.of('result') == []

    def test_no_process_id_refuses_before_running_the_sim(
            self, package, monkeypatch):
        """TASK-2692 (W5): the anuga tool is fail-closed like terrain now.

        Until W5 this asymmetry was deliberate — anuga's terminal channel
        (/process-result/ + /error/) was NOT tombstoned, so a run with no
        Process uuid still landed its result. W5 deletes that channel, so a
        run with no Process uuid has nowhere to report AT ALL. It must refuse
        BEFORE run_sim rather than compute for hours and bin the answer.
        """
        monkeypatch.setenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', 'test-token')
        monkeypatch.delenv('HYDRATA_PROCESS_ID', raising=False)
        monkeypatch.delenv(ALLOW_UNREPORTED_ENV, raising=False)
        mock_run_sim = mock.MagicMock(return_value=None)
        patches = self._patches(mock_run_sim)
        with patches[0], patches[1], patches[2]:
            with pytest.raises(RuntimeError, match='HYDRATA_PROCESS_ID is not set'):
                run_and_report(package, result_bucket='bucket')
        mock_run_sim.assert_not_called()

    def test_opt_out_runs_the_sim_with_no_channel_at_all(
            self, package, monkeypatch):
        """The escape hatch is honest: the sim runs, nothing is reported."""
        monkeypatch.setenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', 'test-token')
        monkeypatch.delenv('HYDRATA_PROCESS_ID', raising=False)
        monkeypatch.setenv(ALLOW_UNREPORTED_ENV, '1')
        mock_run_sim = mock.MagicMock(return_value=None)
        patches = self._patches(mock_run_sim)
        with patches[0], patches[1], patches[2]:
            out = run_and_report(package, result_bucket='bucket')
        mock_run_sim.assert_called_once()
        wrapper = mock_run_sim.call_args.kwargs['callback']
        assert wrapper._inner is None, (
            'no telemetry client -> no web reporter; the deleted '
            'HydrataCallback must not come back'
        )
        assert mock_run_sim.call_args.kwargs['telemetry_client'] is None
        assert out['result_key'] == '601_384_1243_results.zip'
        # NOTHING was reported: there is no second dialect to pick it up.
        assert out['process_result_status'] is None


# ---------------------------------------------------------------------------
# setup_logger — rank gating + telemetry log handler
# ---------------------------------------------------------------------------

def _input_data(tmp_path):
    return {
        'output_directory': str(tmp_path),
        'scenario_config': {
            'control_server': 'https://hydrata.com/',
            'run_id': 1243,
        },
    }


class TestSetupLoggerEventsAndRankGating:
    def _handlers(self, kind):
        from run_anuga import run_utils
        return [h for h in run_utils.logger.handlers if isinstance(h, kind)]

    def test_rank_nonzero_installs_no_web_handler(self, tmp_path, monkeypatch):
        from run_anuga import run_utils

        monkeypatch.setenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', 'tok')
        monkeypatch.setenv('OMPI_COMM_WORLD_RANK', '3')
        run_utils.setup_logger(_input_data(tmp_path), batch_number=1)
        assert self._handlers(run_utils._TelemetryLogHandler) == []

    def test_rank_zero_with_client_ships_log_events(self, tmp_path, monkeypatch):
        from run_anuga import run_utils

        monkeypatch.delenv('OMPI_COMM_WORLD_RANK', raising=False)
        client = FakeTelemetryClient('http://cs', 'p', 't')
        lg = run_utils.setup_logger(
            _input_data(tmp_path), batch_number=1, telemetry_client=client)
        try:
            assert len(self._handlers(run_utils._TelemetryLogHandler)) == 1
            lg.info('run_sim started with batch_number=1')
            logged = [c for c in client.calls if c[0] == 'log']
            assert len(logged) == 1
            assert 'run_sim started' in logged[0][1]
        finally:
            run_utils.setup_logger(_input_data(tmp_path), batch_number=1)

    def test_rank_zero_without_client_installs_no_web_handler(
            self, tmp_path, monkeypatch):
        """TASK-2681: there is no legacy handler left to fall back to.

        A token alone used to arm ``_V2LogHandler`` (POSTing to the now-410
        /log/ route). With one dialect, no client means no web log channel —
        file/console only, which is the honest degrade.
        """
        from run_anuga import run_utils

        monkeypatch.delenv('OMPI_COMM_WORLD_RANK', raising=False)
        monkeypatch.setenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', 'tok')
        run_utils.setup_logger(_input_data(tmp_path), batch_number=1)
        assert self._handlers(run_utils._TelemetryLogHandler) == []
        assert not hasattr(run_utils, '_V2LogHandler'), (
            'the legacy /log/ handler must stay deleted'
        )


# ---------------------------------------------------------------------------
# D7 — no container ETA
# ---------------------------------------------------------------------------

class TestNoContainerEta:
    def test_run_py_carries_no_eta_math(self):
        """AC3 pin: the elapsed-ratio ETA computation is DELETED from run.py
        (server derives ETA from progress history — D7). Source-level pin on
        purpose: the computation must not exist, not merely be unused."""
        import run_anuga.run as run_mod

        source = Path(run_mod.__file__).read_text()
        assert 'eta_seconds = int(elapsed' not in source
        assert 'simulation_start =' not in source  # the ETA wall-clock anchor
        assert 'callback.on_progress(percentage_done)' in source


# ---------------------------------------------------------------------------
# CLI telemetry-log visibility (TASK-2672 review pass)
# ---------------------------------------------------------------------------

class TestCliTelemetryLogging:
    def _stderr_handlers(self, name):
        import logging
        return [
            h for h in logging.getLogger(name).handlers
            if isinstance(h, logging.StreamHandler)
            and getattr(h, 'stream', None) is sys.stderr
        ]

    def test_ensure_cli_logging_attaches_once(self):
        """The fail-loud arming line ('telemetry client active …') is INFO on
        loggers with no handler in a bare container process — python drops
        it. _ensure_cli_logging makes both namespaces stderr-visible, and is
        idempotent (main() may be re-entered in tests)."""
        from run_anuga import cli

        cli._ensure_cli_logging()
        cli._ensure_cli_logging()
        for name in ('run_anuga._handoff', 'gn_anuga.batch_common'):
            handlers = self._stderr_handlers(name)
            assert len(handlers) == 1, (name, handlers)
            import logging
            assert logging.getLogger(name).level <= logging.INFO
