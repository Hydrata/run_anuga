"""TASK-3396 (epic 3221 W2b) — run_and_report and the Result envelope.

The box names a Result envelope at submit (``RESULT_ENVELOPE_S3_URI``) and its
Batch collector reads it when the result POST never lands (TASK-3395). So:

* AC5 — envelope PUT succeeded + POST undelivered: log it, send NO error event,
  return success (the container exits 0 and Batch reports SUCCEEDED). This is
  the only way a lane simulation, which has no inbound path, can succeed.
* AC6 — envelope PUT failed (or none configured) + POST undelivered: raise and
  send the error event, exactly as before.
* AC7 — the ANUGA construction site passes RESULT_ENVELOPE_S3_URI through to
  the TelemetryClient.

Selected by ``-k result_envelope`` (this module's name). Uses the stubbed
``gn_anuga.batch_common`` of test_events_dialect (the real client is baked
into the images; its envelope PUT is proven in hydrata's
tests/unit/test_gn_anuga/test_result_envelope.py).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from run_anuga import _handoff
from run_anuga._handoff import run_and_report

from tests.test_events_dialect import (  # noqa: F401 — _clean_listener is an autouse fixture
    FakeTelemetryClient,
    _capture_client,
    _clean_listener,
    _install_stub_batch_common,
)

URI = 's3://anuga-test-storage/batch-results/proc-uuid-3396/result.json'
#: The wire name the box sets (hydrata telemetry_protocol.RESULT_ENVELOPE_ENV),
#: spelled by value: run_anuga cannot import hydrata in its own CI.
RESULT_ENVELOPE_ENV = 'RESULT_ENVELOPE_S3_URI'


@pytest.fixture
def package(tmp_path: Path) -> Path:
    """The same minimal unzipped package as test_events_dialect's fixture."""
    config = {'id': 384, 'project': 601, 'run_id': 1243,
              'control_server': 'https://hydrata.com/'}
    (tmp_path / 'scenario.json').write_text(json.dumps(config))
    outputs = tmp_path / 'outputs_601_384_1243'
    outputs.mkdir()
    (outputs / 'result_depth_max.tif').write_bytes(b'fake tif')
    return tmp_path


class EnvelopeFakeClient(FakeTelemetryClient):
    """FakeTelemetryClient that also models the envelope outcome."""

    envelope_outcome = True   # what result() reports for the envelope PUT
    post_outcome = False      # what result() reports for the POST

    def __init__(self, *args, result_envelope_uri=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.result_envelope_uri = result_envelope_uri
        self.result_envelope_written = None

    def result(self, **fields):
        self.calls.append(('result', fields))
        if self.result_envelope_uri:
            self.result_envelope_written = self.envelope_outcome
        return self.post_outcome


@pytest.fixture
def envelope_env(monkeypatch):
    monkeypatch.setenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', 'test-token')
    monkeypatch.setenv('HYDRATA_PROCESS_ID', 'proc-uuid-3396')
    monkeypatch.setenv(RESULT_ENVELOPE_ENV, URI)
    client_mod = _install_stub_batch_common(monkeypatch)
    client_mod.TelemetryClient = EnvelopeFakeClient
    return client_mod


def _run(package_dir):
    mock_run_sim = mock.MagicMock(return_value=None)
    with (
        mock.patch('run_anuga.run.run_sim', mock_run_sim),
        mock.patch.object(_handoff, 'upload_cold_archive'),
        mock.patch.object(_handoff, 'upload_result_to_s3'),
    ):
        out = run_and_report(package_dir, result_bucket='bucket')
    return out, _capture_client(mock_run_sim)


def test_result_envelope_written_and_post_undelivered_exits_success(
        package, envelope_env, monkeypatch, caplog):
    monkeypatch.setattr(EnvelopeFakeClient, 'envelope_outcome', True)
    out, client = _run(package)
    assert out == {'result_key': '601_384_1243_results.zip',
                   'process_result_status': 'envelope'}
    assert client.of('error') == []           # no error event: the run is safe
    assert len(client.of('result')) == 1
    assert URI in caplog.text                  # logged, naming the envelope
    assert client.names()[-1] == 'stop_watchdog'


def test_result_envelope_put_failed_and_post_undelivered_raises_as_today(
        package, envelope_env, monkeypatch):
    monkeypatch.setattr(EnvelopeFakeClient, 'envelope_outcome', False)
    mock_run_sim = mock.MagicMock(return_value=None)
    with (
        mock.patch('run_anuga.run.run_sim', mock_run_sim),
        mock.patch.object(_handoff, 'upload_cold_archive'),
        mock.patch.object(_handoff, 'upload_result_to_s3'),
    ):
        with pytest.raises(RuntimeError, match='s3://bucket/601_384_1243_results.zip'):
            run_and_report(package, result_bucket='bucket')
    client = _capture_client(mock_run_sim)
    assert len(client.of('error')) == 1


def test_result_envelope_not_configured_and_post_undelivered_raises_as_today(
        package, envelope_env, monkeypatch):
    monkeypatch.delenv(RESULT_ENVELOPE_ENV)
    mock_run_sim = mock.MagicMock(return_value=None)
    with (
        mock.patch('run_anuga.run.run_sim', mock_run_sim),
        mock.patch.object(_handoff, 'upload_cold_archive'),
        mock.patch.object(_handoff, 'upload_result_to_s3'),
    ):
        with pytest.raises(RuntimeError):
            run_and_report(package, result_bucket='bucket')
    client = _capture_client(mock_run_sim)
    assert client.result_envelope_uri is None
    assert len(client.of('error')) == 1


def test_result_envelope_post_delivered_is_unchanged(package, envelope_env, monkeypatch):
    monkeypatch.setattr(EnvelopeFakeClient, 'post_outcome', True)
    out, client = _run(package)
    assert out['process_result_status'] == 'events'
    assert client.of('error') == []


@pytest.mark.parametrize('value,expected', [(URI, URI), (None, None), ('', None)])
def test_result_envelope_uri_passed_through_to_the_client(
        monkeypatch, envelope_env, value, expected):
    if value is None:
        monkeypatch.delenv(RESULT_ENVELOPE_ENV)
    else:
        monkeypatch.setenv(RESULT_ENVELOPE_ENV, value)
    client = _handoff._make_telemetry_client({'control_server': 'https://cs'})
    assert isinstance(client, EnvelopeFakeClient)
    assert client.result_envelope_uri == expected


def test_result_envelope_env_name_matches_the_box_contract():
    """_handoff's constant is the box's wire name, pinned by value."""
    assert getattr(_handoff, 'RESULT_ENVELOPE_ENV', None) == RESULT_ENVELOPE_ENV


class OldSignatureClient(FakeTelemetryClient):
    """A pre-3396 TelemetryClient: the OLD constructor, no **kwargs, so an
    unexpected ``result_envelope_uri=`` keyword is a TypeError (an image whose
    staged hydrata leaf predates the envelope writer)."""

    def __init__(self, control_server, process_id, token, *, timeout_s=10):
        super().__init__(control_server, process_id, token)
        self.timeout_s = timeout_s


@pytest.mark.parametrize('value', [None, ''])
def test_result_envelope_unset_constructs_an_old_signature_client(
        monkeypatch, envelope_env, value):
    """F5 (compat): with RESULT_ENVELOPE_S3_URI unset or empty, the envelope
    keyword is not passed at all, so an old-signature client still builds."""
    if value is None:
        monkeypatch.delenv(RESULT_ENVELOPE_ENV)
    else:
        monkeypatch.setenv(RESULT_ENVELOPE_ENV, value)
    envelope_env.TelemetryClient = OldSignatureClient
    client = _handoff._make_telemetry_client({'control_server': 'https://cs'})
    assert isinstance(client, OldSignatureClient)
    assert not hasattr(client, 'result_envelope_uri')
