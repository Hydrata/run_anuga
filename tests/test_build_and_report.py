"""TASK-3231 (epic 3221 W2b) — the ``build-and-report`` verb: a mesh build on AWS Batch.

The web box stages a scenario package WITHOUT the mesh (scenario.json +
inputs/) and submits an ``anuga-build`` job. The job runs this verb: mesh with
the SAME ``create_anuga_mesh`` the box and ``run_sim`` use, package the whole
directory (the ``.msh`` included, so the GPU job reuses it and never re-meshes:
``run_sim`` skips ``create_anuga_mesh`` when ``input_data['mesh_filepath']``
exists), upload it to the key the box named, and report a terminal ``result``
event through the TelemetryClient — which writes the Result envelope first
(TASK-3396), so the box's Batch collector can complete the build with no POST
(the non-prod lane has no inbound path).

Selected by ``-k build_and_report`` (every test name carries it). Uses the
stubbed ``gn_anuga.batch_common`` of test_events_dialect.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from run_anuga import _handoff

from tests.test_events_dialect import (  # noqa: F401 — _clean_listener is an autouse fixture
    FakeTelemetryClient,
    _clean_listener,
    _install_stub_batch_common,
)

FIXTURE = Path(__file__).resolve().parent / 'data' / 'minimal_package'
URI = 's3://anuga-test-storage/batch-results/proc-build/disp/result.json'
RESULT_ENVELOPE_ENV = 'RESULT_ENVELOPE_S3_URI'
BUILT_KEY = 'anuga_build/1/abc123/run_1_1_1_built.zip'


class BuildFakeClient(FakeTelemetryClient):
    """FakeTelemetryClient that also models the envelope PUT outcome."""

    instances: list = []
    envelope_outcome = True
    post_outcome = True

    def __init__(self, *args, result_envelope_uri=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.result_envelope_uri = result_envelope_uri
        self.result_envelope_written = None
        BuildFakeClient.instances.append(self)

    def result(self, **fields):
        self.calls.append(('result', fields))
        if self.result_envelope_uri:
            self.result_envelope_written = self.envelope_outcome
        return self.post_outcome


@pytest.fixture
def package(tmp_path: Path) -> Path:
    """The minimal fixture package as the box stages it: no mesh yet."""
    pkg = tmp_path / 'pkg'
    shutil.copytree(FIXTURE, pkg)
    (pkg / 'outputs_1_1_1' / 'run_1_1_1.msh').unlink()
    return pkg


@pytest.fixture
def build_env(monkeypatch):
    BuildFakeClient.instances = []
    monkeypatch.setattr(BuildFakeClient, 'envelope_outcome', True)
    monkeypatch.setattr(BuildFakeClient, 'post_outcome', True)
    monkeypatch.setenv('HYDRATA_INTERNAL_COMPUTE_TOKEN', 'test-token')
    monkeypatch.setenv('HYDRATA_PROCESS_ID', 'proc-build')
    monkeypatch.setenv(RESULT_ENVELOPE_ENV, URI)
    monkeypatch.setenv('PACKAGE_S3_BUCKET', 'anuga-run-bucket')
    monkeypatch.setenv('BUILT_PACKAGE_S3_KEY', BUILT_KEY)
    client_mod = _install_stub_batch_common(monkeypatch)
    client_mod.TelemetryClient = BuildFakeClient
    return client_mod


class _UploadSpy:
    """Captures the uploaded zip's member names (the zip is deleted after)."""

    def __init__(self):
        self.calls = []

    def __call__(self, zip_path, bucket, key):
        with zipfile.ZipFile(zip_path) as zf:
            self.calls.append((bucket, key, sorted(zf.namelist())))


def test_build_and_report_meshes_packages_the_msh_and_reports_the_result(package, build_env):
    spy = _UploadSpy()
    with mock.patch.object(_handoff, 'upload_result_to_s3', spy):
        out = _handoff.build_and_report(package)

    assert out['built_package_key'] == BUILT_KEY
    assert out['process_result_status'] == 'events'
    # ONE upload, to the key and bucket the box named, carrying the mesh the
    # GPU job will reuse plus the package it came from.
    assert len(spy.calls) == 1
    bucket, key, names = spy.calls[0]
    assert (bucket, key) == ('anuga-run-bucket', BUILT_KEY)
    assert 'outputs_1_1_1/run_1_1_1.msh' in names
    assert 'scenario.json' in names and 'inputs/boundary.geojson' in names
    client = BuildFakeClient.instances[-1]
    assert client.result_envelope_uri == URI
    assert client.names()[0] == 'started'
    assert client.of('error') == []
    (_, fields), = client.of('result')
    assert fields['built_package_key'] == BUILT_KEY
    assert fields['mesh_triangle_count'] == out['mesh_triangle_count'] > 0
    assert fields['mesh_node_count'] > 0
    assert fields['mesh_qa']['triangle_count'] == fields['mesh_triangle_count']
    json.dumps(fields)  # the event body must be JSON-serialisable
    assert client.names()[-1] == 'stop_watchdog'


def test_build_and_report_post_undelivered_but_envelope_written_exits_success(package, build_env, caplog):
    """The lane shape: no inbound path, so the POST never lands; the envelope
    did, so the job must exit 0 (the collector completes it) and send no error."""
    BuildFakeClient.post_outcome = False
    with mock.patch.object(_handoff, 'upload_result_to_s3', _UploadSpy()):
        out = _handoff.build_and_report(package)
    assert out['process_result_status'] == 'envelope'
    client = BuildFakeClient.instances[-1]
    assert client.of('error') == []
    assert URI in caplog.text


def test_build_and_report_post_and_envelope_both_failed_raises(package, build_env):
    BuildFakeClient.post_outcome = False
    BuildFakeClient.envelope_outcome = False
    with mock.patch.object(_handoff, 'upload_result_to_s3', _UploadSpy()):
        with pytest.raises(RuntimeError, match=BUILT_KEY):
            _handoff.build_and_report(package)
    client = BuildFakeClient.instances[-1]
    assert len(client.of('error')) == 1


def test_build_and_report_mesh_failure_reports_an_error_without_traceback(package, build_env):
    upload = mock.MagicMock()
    with mock.patch('run_anuga.run_utils.create_anuga_mesh',
                    side_effect=ValueError('boundary polygon self-intersects')), \
         mock.patch.object(_handoff, 'upload_result_to_s3', upload):
        with pytest.raises(ValueError, match='self-intersects'):
            _handoff.build_and_report(package)
    upload.assert_not_called()
    client = BuildFakeClient.instances[-1]
    (_, message), = client.of('error')
    assert message == 'ValueError: boundary polygon self-intersects'
    assert 'Traceback' not in message
    assert client.of('result') == []


def test_build_and_report_refuses_without_a_process_id(package, build_env, monkeypatch):
    """Fail-closed like run-and-report: no channel, no build."""
    monkeypatch.delenv('HYDRATA_PROCESS_ID')
    mesher = mock.MagicMock()
    with mock.patch('run_anuga.run_utils.create_anuga_mesh', mesher):
        with pytest.raises(RuntimeError, match='HYDRATA_PROCESS_ID'):
            _handoff.build_and_report(package)
    mesher.assert_not_called()


def test_build_and_report_refuses_without_a_built_package_key(package, build_env, monkeypatch):
    monkeypatch.delenv('BUILT_PACKAGE_S3_KEY')
    mesher = mock.MagicMock()
    with mock.patch('run_anuga.run_utils.create_anuga_mesh', mesher):
        with pytest.raises(RuntimeError, match='BUILT_PACKAGE_S3_KEY'):
            _handoff.build_and_report(package)
    mesher.assert_not_called()


def test_build_and_report_cli_verb_is_wired(package):
    from run_anuga import cli

    with mock.patch('run_anuga._handoff.build_and_report',
                    return_value={'built_package_key': BUILT_KEY, 'mesh_triangle_count': 3,
                                  'process_result_status': 'events'}) as verb, \
         mock.patch('sys.argv', ['run-anuga', 'build-and-report', str(package),
                                 '--package-key', BUILT_KEY, '--package-bucket', 'b']):
        cli.main()
    verb.assert_called_once_with(str(package), package_bucket='b', package_key=BUILT_KEY)


# ---------------------------------------------------------------------------
# The Batch entrypoint selects the verb (ANUGA_VERB, allow-listed)
# ---------------------------------------------------------------------------

def test_build_and_report_entrypoint_verb_runs_single_process(tmp_path):
    from tests.test_anuga_entrypoint_error_trap import _run_entrypoint

    proc, cap = _run_entrypoint(tmp_path, process_id=None, aws_fails=False,
                                extra_env={'ANUGA_VERB': 'build-and-report', 'CPUS': '4'})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert 'fake python: -m run_anuga.cli build-and-report /tmp/simulation' in proc.stdout
    assert 'mpirun' not in proc.stdout


def test_build_and_report_entrypoint_defaults_to_run_and_report(tmp_path):
    from tests.test_anuga_entrypoint_error_trap import _run_entrypoint

    proc, _ = _run_entrypoint(tmp_path, process_id=None, aws_fails=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert 'fake python: -m run_anuga.cli run-and-report /tmp/simulation' in proc.stdout


def test_build_and_report_entrypoint_unknown_verb_exits_2_before_download(tmp_path):
    from tests.test_anuga_entrypoint_error_trap import _run_entrypoint

    proc, cap = _run_entrypoint(tmp_path, process_id=None, aws_fails=False,
                                extra_env={'ANUGA_VERB': 'rm -rf'})
    assert proc.returncode == 2
    assert 'unknown ANUGA_VERB' in proc.stderr
    assert 'Downloading simulation package' not in proc.stdout
    assert not cap
