"""Tests for run_anuga._handoff — TASK-1159 (F1) result handoff.

Covers:

* The shared field-name constant (``RESULT_PACKAGE_KEY_FIELD``).
* ``make_result_key`` matches the entrypoint.sh shape.
* ``zip_outputs`` respects the entrypoint's exclusion list (``package.zip``,
  the result zip itself, and any embedded ``run_anuga/`` source tree).
* ``upload_result_to_s3`` routes through the hardened ``_make_s3_client()``
  factory and calls ``upload_file`` with an explicit ``TransferConfig``
  (TASK-2033: was bare ``boto3.client("s3")`` → S3 IncompleteBody in run-1266).
* ``run_and_report`` orchestrates run_sim + zip + upload + the terminal
  ``result`` event, and posts a terminal ``error`` event on failure (the F0
  wedge defence).
* The module imports cleanly without Django (run_anuga must stay Django-free).
* TASK-2033: ``_make_s3_client`` factory uses
  ``request_checksum_calculation='when_required'`` so both upload paths are
  immune to the aws-chunked UploadPart IncompleteBody regression.

TASK-2692 (epic 2662 W5) deleted ``report_result`` / ``report_error`` and their
``/api/v2/anuga/runs/<id>/{process-result,error}/`` routes (410 tombstones), so
the tests that pinned their wire shape are gone with them and the orchestration
tests below read the terminal event off the events client instead. That also
makes ``run_and_report`` fail-closed, which is why every test here that reaches
it arms the ``events_env`` fixture.
"""

from __future__ import annotations

import contextlib
import json
import sys
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from run_anuga._handoff import (
    ALLOW_UNREPORTED_ENV,
    COLD_ARCHIVE_PREFIX_FIELD,
    PLAYBACK_STORE_PREFIX_FIELD,
    RESULT_PACKAGE_KEY_FIELD,
    make_cold_archive_prefix,
    make_result_key,
    run_and_report,
    upload_cold_archive,
    zip_outputs,
)

# The events-dialect test double lives with the tests that own it; reusing it
# here keeps ONE stub client in the suite rather than a second, drifting copy.
from tests.test_events_dialect import (  # noqa: E402
    FakeTelemetryClient,
    _install_stub_batch_common,
)


@pytest.fixture
def events_env(monkeypatch):
    """Arm the events dialect so ``run_and_report`` will run at all.

    TASK-2692 (W5) made ``run_and_report`` fail-closed: with no
    ``HYDRATA_PROCESS_ID`` and no importable ``gn_anuga.batch_common`` it
    refuses before the sim. The orchestration tests below are about the
    handoff, not about the construction site (that is
    ``test_events_dialect.TestMakeTelemetryClient``'s job), so they arm the
    same stub client and read the terminal event off it.
    """
    monkeypatch.setenv("HYDRATA_INTERNAL_COMPUTE_TOKEN", "test-token")
    monkeypatch.setenv("HYDRATA_PROCESS_ID", "proc-uuid-2692")
    monkeypatch.delenv(ALLOW_UNREPORTED_ENV, raising=False)
    _install_stub_batch_common(monkeypatch)


def _captured_client(mock_run_sim) -> FakeTelemetryClient:
    """The telemetry client ``run_and_report`` built and handed to run_sim."""
    return mock_run_sim.call_args.kwargs["telemetry_client"]


def _result_event_fields(mock_run_sim) -> dict:
    """The single terminal ``result`` event's payload fields."""
    events = _captured_client(mock_run_sim).of("result")
    assert len(events) == 1, events
    return events[0][1]


def _handoff_module_path() -> str:
    from run_anuga import _handoff

    return _handoff.__file__


def test_shared_constant_value():
    """The constant's value must match what gn_anuga.api_v2.process_result reads."""
    assert RESULT_PACKAGE_KEY_FIELD == "result_package_key"


def test_module_imports_without_django():
    """run_anuga must stay Django-free; import the module in a fresh subprocess."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", "import run_anuga._handoff"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
    )
    assert result.returncode == 0, (
        f"run_anuga._handoff failed to import standalone:\nstderr: {result.stderr}"
    )


def test_make_result_key_format():
    """Mirrors batch/entrypoint.sh line 34."""
    assert make_result_key(601, 384, 1243) == "601_384_1243_results.zip"


class TestZipOutputs:
    def test_includes_top_level_file(self, tmp_path: Path):
        (tmp_path / "scenario.json").write_text("{}")
        zip_path = tmp_path / "result.zip"
        zip_outputs(tmp_path, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            assert "scenario.json" in zf.namelist()

    def test_includes_nested_outputs(self, tmp_path: Path):
        outputs = tmp_path / "outputs_1_1_1"
        outputs.mkdir()
        (outputs / "result_depth_max.tif").write_bytes(b"fake tif")
        zip_path = tmp_path / "result.zip"
        zip_outputs(tmp_path, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            assert "outputs_1_1_1/result_depth_max.tif" in zf.namelist()

    def test_excludes_package_zip(self, tmp_path: Path):
        (tmp_path / "package.zip").write_bytes(b"input zip should not be re-zipped")
        (tmp_path / "scenario.json").write_text("{}")
        zip_path = tmp_path / "result.zip"
        zip_outputs(tmp_path, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            assert "package.zip" not in zf.namelist()

    def test_excludes_run_anuga_source_tree(self, tmp_path: Path):
        ra = tmp_path / "run_anuga"
        ra.mkdir()
        (ra / "cli.py").write_text("# embedded source")
        (tmp_path / "scenario.json").write_text("{}")
        zip_path = tmp_path / "result.zip"
        zip_outputs(tmp_path, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            assert "run_anuga/cli.py" not in zf.namelist()

    def test_excludes_self(self, tmp_path: Path):
        (tmp_path / "scenario.json").write_text("{}")
        zip_path = tmp_path / "601_384_1243_results.zip"
        zip_outputs(tmp_path, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            assert "601_384_1243_results.zip" not in zf.namelist()

    def _make_realistic_outputs(self, tmp_path: Path) -> Path:
        """A package mirroring a real ANUGA output dir (TASK-1821 slimming)."""
        (tmp_path / "scenario.json").write_text("{}")
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        (inputs / "ele_merewether_dem.tif").write_bytes(b"input dem")
        outputs = tmp_path / "outputs_601_384_1243"
        outputs.mkdir()
        # Kept: the BE payload + provenance.
        (outputs / "run_601_384_1243_depth_max.tif").write_bytes(b"depth max")
        (outputs / "run_601_384_1243_velocity_max.tif").write_bytes(b"vel max")
        (outputs / "run_601_384_1243_depthIntegratedVelocity_max.tif").write_bytes(b"div max")
        (outputs / "run_anuga_1.log").write_text("log line")
        # Excluded: raw sww, per-timestep rasters, mesh, MPI checkpoints.
        (outputs / "run_601_384_1243.sww").write_bytes(b"huge netcdf" * 1000)
        (outputs / "run_601_384_1243_depth_0_Time_0.tif").write_bytes(b"t0")
        (outputs / "run_601_384_1243_velocity_5_Time_300.tif").write_bytes(b"t5")
        (outputs / "run_601_384_1243.msh").write_bytes(b"mesh")
        checkpoints = outputs / "checkpoints"
        checkpoints.mkdir()
        (checkpoints / "run_601_384_1243_P32_8_0.0.pickle").write_bytes(b"ckpt" * 1000)
        return outputs

    def test_slim_keeps_only_max_tifs_and_provenance(self, tmp_path: Path):
        self._make_realistic_outputs(tmp_path)
        zip_path = tmp_path / "601_384_1243_results.zip"
        zip_outputs(tmp_path, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())

        # The 3 max rasters that Run.process_result extracts MUST survive.
        assert "outputs_601_384_1243/run_601_384_1243_depth_max.tif" in names
        assert "outputs_601_384_1243/run_601_384_1243_velocity_max.tif" in names
        assert "outputs_601_384_1243/run_601_384_1243_depthIntegratedVelocity_max.tif" in names
        # Provenance kept.
        assert "scenario.json" in names
        assert "outputs_601_384_1243/run_anuga_1.log" in names
        assert "inputs/ele_merewether_dem.tif" in names

        # The bulk / dead-weight artifacts are gone.
        assert not any(n.endswith(".sww") for n in names)
        assert not any("_Time_" in n for n in names)
        assert not any(n.endswith(".msh") for n in names)
        assert not any("/checkpoints/" in n for n in names)

    def test_slim_excludes_sww(self, tmp_path: Path):
        self._make_realistic_outputs(tmp_path)
        zip_path = tmp_path / "result.zip"
        zip_outputs(tmp_path, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            assert "outputs_601_384_1243/run_601_384_1243.sww" not in zf.namelist()

    def test_slim_excludes_checkpoints_subtree(self, tmp_path: Path):
        self._make_realistic_outputs(tmp_path)
        zip_path = tmp_path / "result.zip"
        zip_outputs(tmp_path, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            assert not any("checkpoints" in n for n in zf.namelist())

    def test_slim_excludes_per_timestep_tifs_but_keeps_max(self, tmp_path: Path):
        self._make_realistic_outputs(tmp_path)
        zip_path = tmp_path / "result.zip"
        zip_outputs(tmp_path, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
        assert not any("_Time_" in n for n in names)
        assert any(n.endswith("_depth_max.tif") for n in names)


class TestUploadResultToS3:
    def test_uploads_via_boto3_client(self, tmp_path: Path):
        """upload_result_to_s3 uses _make_s3_client and calls upload_file with right args.

        After TASK-2033 the function routes through _make_s3_client() (hardened factory)
        and passes an explicit TransferConfig — the factory is patched here so the test
        does not need real boto3/botocore or AWS credentials.
        """
        zip_path = tmp_path / "result.zip"
        zip_path.write_bytes(b"zip")
        from run_anuga import _handoff

        mock_s3 = mock.MagicMock()
        with mock.patch.object(_handoff, "_make_s3_client", return_value=mock_s3), \
             mock.patch.object(_handoff, "import_optional"):  # covers boto3.s3.transfer
            _handoff.upload_result_to_s3(zip_path, "test-bucket", "601_384_1243_results.zip")

        mock_s3.upload_file.assert_called_once()
        # Check the positional args; Config= kwarg (TransferConfig) is present but not
        # asserted here — its correctness is covered by TestHardenedS3ClientFactory.
        upload_pos_args = mock_s3.upload_file.call_args.args
        assert upload_pos_args == (str(zip_path), "test-bucket", "601_384_1243_results.zip")


class TestLegacyReportersAreGone:
    """TASK-2692 (epic 2662 W5, AC3) — the legacy terminal dialect is DELETED.

    ``TestReportResult`` / ``TestReportError`` used to live here, pinning the
    wire shape of ``POST /api/v2/anuga/runs/<id>/{process-result,error}/``.
    Both routes are 410 tombstones now. These pins replace them so a future
    "restore the fallback" patch fails loudly instead of quietly re-creating a
    caller of a dead route.
    """

    def test_report_result_and_report_error_no_longer_exist(self):
        from run_anuga import _handoff

        assert not hasattr(_handoff, "report_result")
        assert not hasattr(_handoff, "report_error")

    def test_handoff_builds_no_legacy_run_scoped_url(self):
        """A URL is BUILT by interpolating the run id — ban exactly that shape.

        Prose may still NAME the dead routes when explaining the deletion; what
        must never come back is a live ``anuga/runs/{...}/`` URL.
        """
        source = Path(_handoff_module_path()).read_text()
        assert "anuga/runs/{" not in source, (
            "run_anuga/_handoff.py builds a legacy /api/v2/anuga/runs/<id>/ URL "
            "again; those routes are 410 tombstones since TASK-2692"
        )


class TestRunAndReportOrchestration:
    @pytest.fixture
    def package(self, tmp_path: Path) -> Path:
        config = {
            "id": 384,
            "project": 601,
            "run_id": 1243,
            "control_server": "https://hydrata.com/",
        }
        (tmp_path / "scenario.json").write_text(json.dumps(config))
        outputs = tmp_path / "outputs_601_384_1243"
        outputs.mkdir()
        (outputs / "result_depth_max.tif").write_bytes(b"fake tif")
        return tmp_path

    def test_required_token_env_var_validated(self, package: Path, monkeypatch):
        """Missing HYDRATA_INTERNAL_COMPUTE_TOKEN raises before anything ships."""
        monkeypatch.delenv("HYDRATA_INTERNAL_COMPUTE_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="HYDRATA_INTERNAL_COMPUTE_TOKEN"):
            run_and_report(package, result_bucket="bucket")

    def test_required_result_bucket_env_var_validated(self, package: Path, monkeypatch):
        """Missing RESULT_S3_BUCKET (and no result_bucket kwarg) raises BEFORE run_sim.

        Fail-fast: a misconfigured worker mustn't burn N hours of ANUGA compute
        before discovering it has nowhere to upload the result.
        """
        monkeypatch.setenv("HYDRATA_INTERNAL_COMPUTE_TOKEN", "test-token")
        monkeypatch.delenv("RESULT_S3_BUCKET", raising=False)

        with mock.patch("run_anuga.run.run_sim") as mock_run_sim:
            with pytest.raises(RuntimeError, match="RESULT_S3_BUCKET"):
                run_and_report(package)
        mock_run_sim.assert_not_called()

    def test_happy_path_zips_uploads_and_posts(self, package: Path, events_env):
        from run_anuga import _handoff

        mock_run_sim = mock.MagicMock(return_value=None)
        # `run_sim` is imported lazily inside run_and_report (`from run_anuga.run
        # import run_sim`) so patch the source module, not _handoff.
        with mock.patch("run_anuga.run.run_sim", mock_run_sim), \
             mock.patch.object(_handoff, "upload_cold_archive") as mock_archive, \
             mock.patch.object(_handoff, "upload_result_to_s3") as mock_upload:
            result = run_and_report(package, result_bucket="bucket")

        mock_run_sim.assert_called_once()
        mock_archive.assert_called_once()
        mock_upload.assert_called_once()
        upload_args = mock_upload.call_args[0]
        assert upload_args[1] == "bucket"
        assert upload_args[2] == "601_384_1243_results.zip"
        # W2 (TASK-1920): the result event carries cold_archive_prefix.
        # TASK-2623: playback_store_prefix is OMITTED here since this fixture's
        # package has no playback_store_export_result.json marker (run_sim is
        # mocked out, so TASK-2622's exporter never ran) — never a fabricated
        # null.
        assert _result_event_fields(mock_run_sim) == {
            RESULT_PACKAGE_KEY_FIELD: "601_384_1243_results.zip",
            COLD_ARCHIVE_PREFIX_FIELD: "cold-archive/601_384_1243/",
        }
        # No error event on the happy path — a future regression that reported
        # an error on success would otherwise silently slip through.
        assert _captured_client(mock_run_sim).of("error") == []
        assert result == {"result_key": "601_384_1243_results.zip",
                          "process_result_status": "events"}

    def test_sim_failure_posts_error_then_raises(self, package: Path, events_env):
        from run_anuga import _handoff

        mock_run_sim = mock.MagicMock(side_effect=RuntimeError("sim boom"))
        with mock.patch("run_anuga.run.run_sim", mock_run_sim), \
             mock.patch.object(_handoff, "upload_result_to_s3") as mock_upload:
            with pytest.raises(RuntimeError, match="sim boom"):
                run_and_report(package, result_bucket="bucket")

        mock_upload.assert_not_called()
        errors = _captured_client(mock_run_sim).of("error")
        assert len(errors) == 1 and "sim boom" in errors[0][1]

    def test_undeliverable_result_event_raises(self, package: Path, events_env):
        """TASK-2692 replaces the old "non-2xx /process-result/ -> POST /error/
        -> raise" branch: the route is gone, so an undeliverable terminal
        ``result`` event raises directly (container exits non-zero -> Batch job
        FAILED, which the reaper and an operator can both see)."""
        from run_anuga import _handoff

        mock_run_sim = mock.MagicMock(return_value=None)
        failing = FakeTelemetryClient("http://cs", "proc-uuid-2692", "test-token")
        failing.result_ok = False

        def make_failing(scenario_config):
            return failing

        with mock.patch("run_anuga.run.run_sim", mock_run_sim), \
             mock.patch.object(_handoff, "upload_cold_archive"), \
             mock.patch.object(_handoff, "upload_result_to_s3"), \
             mock.patch.object(_handoff, "_make_telemetry_client", make_failing):
            with pytest.raises(RuntimeError, match="was not accepted"):
                run_and_report(package, result_bucket="bucket")


# ---------------------------------------------------------------------------
# W2 (TASK-1920) — Cold archive tests
# ---------------------------------------------------------------------------

class TestMakeColdArchivePrefix:
    def test_prefix_format(self):
        """Prefix must be distinct from the result key and target cold-archive/."""
        prefix = make_cold_archive_prefix(601, 384, 1243)
        assert prefix == "cold-archive/601_384_1243/"
        assert prefix.startswith("cold-archive/")
        assert prefix.endswith("/")

    def test_prefix_differs_from_result_key(self):
        """Cold archive prefix must not collide with the slim result zip key."""
        result_key = make_result_key(601, 384, 1243)
        prefix = make_cold_archive_prefix(601, 384, 1243)
        assert not prefix.startswith(result_key)
        assert not result_key.startswith("cold-archive/")

    def test_cold_archive_prefix_field_constant(self):
        """Wire-field constant value must match what the BE receiver reads."""
        assert COLD_ARCHIVE_PREFIX_FIELD == "cold_archive_prefix"


class TestUploadColdArchive:
    """upload_cold_archive() uploads the .sww directly (no combined zip) plus
    package.zip, scenario.json, and 3 *_max.tif files via upload_file calls.
    Mirrors the existing mock pattern in TestUploadResultToS3.
    """

    def _make_package(self, tmp_path: Path) -> Path:
        """Create a minimal package directory mirroring a real ANUGA output."""
        run_label = "601_384_1243"
        output_dir = tmp_path / f"outputs_{run_label}"
        output_dir.mkdir()
        (output_dir / f"{run_label}.sww").write_bytes(b"fake sww bytes")
        (output_dir / f"run_{run_label}_depth_max.tif").write_bytes(b"depth")
        (output_dir / f"run_{run_label}_velocity_max.tif").write_bytes(b"velocity")
        (output_dir / f"run_{run_label}_depthIntegratedVelocity_max.tif").write_bytes(b"div")
        (output_dir / "run_diagnostics_1.csv").write_text("# mesh stats\nsim_time_s,min_dt_ms\n60.0,40.0\n")
        (tmp_path / "package.zip").write_bytes(b"input zip")
        (tmp_path / "scenario.json").write_text("{}")
        return tmp_path

    def test_uploads_sww_directly_not_combined_zip(self, tmp_path: Path):
        """The .sww is uploaded via upload_file directly — NO combined zip created."""
        package = self._make_package(tmp_path)
        from run_anuga import _handoff

        with mock.patch.object(_handoff, "import_optional") as mock_import:
            mock_s3 = mock.MagicMock()
            mock_import.return_value.client.return_value = mock_s3
            upload_cold_archive(
                package,
                "test-bucket",
                "cold-archive/601_384_1243/",
                project_id=601,
                scenario_id=384,
                run_id=1243,
            )

        # upload_file called (not put_object or create_multipart_upload directly)
        assert mock_s3.upload_file.called, "upload_file must be called"
        # No combined zip file was created (the pre-seeded package.zip is the only zip)
        import glob
        zip_files = [f for f in glob.glob(str(tmp_path / "*.zip"))
                     if not f.endswith("package.zip")]
        assert len(zip_files) == 0, (
            f"Unexpected combined zip files created: {zip_files}"
        )

    def test_uploads_correct_number_of_objects(self, tmp_path: Path):
        """upload_file called for .sww + package.zip + scenario.json + 3 *_max.tif
        + 1 run_diagnostics_*.csv = 7 (TASK-2622 added the diagnostics CSV so the
        playback-store's min_dt_ms/mean_dt_ms/last_dt_ms survive SWW cold-archive)."""
        package = self._make_package(tmp_path)
        from run_anuga import _handoff

        with mock.patch.object(_handoff, "import_optional") as mock_import:
            mock_s3 = mock.MagicMock()
            mock_import.return_value.client.return_value = mock_s3
            upload_cold_archive(
                package,
                "test-bucket",
                "cold-archive/601_384_1243/",
                project_id=601,
                scenario_id=384,
                run_id=1243,
            )

        call_count = mock_s3.upload_file.call_count
        assert call_count == 7, f"Expected 7 upload_file calls, got {call_count}"

    def test_uploads_diagnostics_csv(self, tmp_path: Path):
        """run_diagnostics_*.csv is uploaded (glob, since multi-batch/checkpoint
        -restart runs can write more than one)."""
        package = self._make_package(tmp_path)
        from run_anuga import _handoff

        with mock.patch.object(_handoff, "import_optional") as mock_import:
            mock_s3 = mock.MagicMock()
            mock_import.return_value.client.return_value = mock_s3
            upload_cold_archive(
                package,
                "test-bucket",
                "cold-archive/601_384_1243/",
                project_id=601,
                scenario_id=384,
                run_id=1243,
            )

        keys = [call.args[2] for call in mock_s3.upload_file.call_args_list]
        assert "cold-archive/601_384_1243/run_diagnostics_1.csv" in keys

    def test_uploads_multiple_diagnostics_csvs(self, tmp_path: Path):
        """A checkpoint-restart / multi-batch run can write several
        run_diagnostics_N.csv files — all of them are archived."""
        package = self._make_package(tmp_path)
        run_label = "601_384_1243"
        (package / f"outputs_{run_label}" / "run_diagnostics_2.csv").write_text(
            "# mesh stats\nsim_time_s,min_dt_ms\n120.0,38.0\n"
        )
        from run_anuga import _handoff

        with mock.patch.object(_handoff, "import_optional") as mock_import:
            mock_s3 = mock.MagicMock()
            mock_import.return_value.client.return_value = mock_s3
            upload_cold_archive(
                package,
                "test-bucket",
                "cold-archive/601_384_1243/",
                project_id=601,
                scenario_id=384,
                run_id=1243,
            )

        keys = {call.args[2] for call in mock_s3.upload_file.call_args_list}
        assert "cold-archive/601_384_1243/run_diagnostics_1.csv" in keys
        assert "cold-archive/601_384_1243/run_diagnostics_2.csv" in keys

    def test_uploads_under_correct_prefix(self, tmp_path: Path):
        """All S3 keys must be under the cold-archive prefix."""
        package = self._make_package(tmp_path)
        from run_anuga import _handoff

        with mock.patch.object(_handoff, "import_optional") as mock_import:
            mock_s3 = mock.MagicMock()
            mock_import.return_value.client.return_value = mock_s3
            upload_cold_archive(
                package,
                "test-bucket",
                "cold-archive/601_384_1243/",
                project_id=601,
                scenario_id=384,
                run_id=1243,
            )

        for call in mock_s3.upload_file.call_args_list:
            _, bucket, key = call.args
            assert bucket == "test-bucket"
            assert key.startswith("cold-archive/601_384_1243/"), (
                f"Key {key!r} not under cold-archive prefix"
            )

    def test_rank0_gate_in_run_and_report(self, tmp_path: Path, monkeypatch):
        """upload_cold_archive is called from run_and_report ONLY on rank 0.

        Non-rank-0 processes return early before the cold-archive step.
        """
        from run_anuga import _handoff

        monkeypatch.setenv("HYDRATA_INTERNAL_COMPUTE_TOKEN", "test-token")
        config = {
            "id": 384, "project": 601, "run_id": 1243,
            "control_server": "https://hydrata.com/",
        }
        (tmp_path / "scenario.json").write_text(json.dumps(config))

        with mock.patch("run_anuga.run.run_sim"), \
             mock.patch.object(_handoff, "_is_mpi_rank_zero", return_value=False), \
             mock.patch.object(_handoff, "upload_cold_archive") as mock_archive, \
             mock.patch.object(_handoff, "upload_result_to_s3"):
            result = run_and_report(tmp_path, result_bucket="bucket")

        # Non-rank-0: returns early, cold archive NOT called.
        mock_archive.assert_not_called()
        assert result["result_key"] is None

    def test_is_mpi_rank_zero_reads_launcher_env(self, monkeypatch):
        """_is_mpi_rank_zero MUST read the launcher rank env (survives MPI_FINALIZE).

        Regression for TASK-1952: anuga's run_sim finalizes MPI *before* the
        handoff runs, so the old ``if MPI.Is_finalized(): return True`` short
        circuit (TASK-1182) made EVERY rank look like rank 0 — under
        ``mpirun -np N`` all N ranks then ran the zip + S3 upload +
        /process-result/ POST concurrently and raced (XAmzContentSHA256Mismatch
        / IncompleteBody on UploadPart, then HTTP 409 'run already terminal').
        Open MPI sets ``OMPI_COMM_WORLD_RANK`` (PMIx sets ``PMIX_RANK``) per rank
        at spawn; those survive finalize and are the authoritative rank source.
        """
        from run_anuga._handoff import _is_mpi_rank_zero

        for var in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK"):
            monkeypatch.delenv(var, raising=False)

        # Non-zero rank under Open MPI -> NOT the handoff rank (the bug: was True).
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "5")
        assert _is_mpi_rank_zero() is False

        # Rank 0 under Open MPI -> the handoff rank.
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
        assert _is_mpi_rank_zero() is True

        # PMIx launcher (Open MPI 5 / Slurm PMIx), non-zero rank.
        monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
        monkeypatch.setenv("PMIX_RANK", "3")
        assert _is_mpi_rank_zero() is False

    def test_is_mpi_rank_zero_single_process_cli(self, monkeypatch):
        """With no launcher env (a localhost CLI run) the lone process IS rank 0."""
        from run_anuga._handoff import _is_mpi_rank_zero

        for var in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK"):
            monkeypatch.delenv(var, raising=False)
        assert _is_mpi_rank_zero() is True

    def test_cold_archive_best_effort_does_not_fail_run(self, tmp_path: Path, events_env):
        """A cold archive failure is logged but must NOT raise out of run_and_report."""
        from run_anuga import _handoff

        config = {
            "id": 384, "project": 601, "run_id": 1243,
            "control_server": "https://hydrata.com/",
        }
        (tmp_path / "scenario.json").write_text(json.dumps(config))
        outputs = tmp_path / "outputs_601_384_1243"
        outputs.mkdir()
        (outputs / "result_depth_max.tif").write_bytes(b"fake tif")

        mock_run_sim = mock.MagicMock(return_value=None)
        with mock.patch("run_anuga.run.run_sim", mock_run_sim), \
             mock.patch.object(_handoff, "upload_cold_archive",
                               side_effect=RuntimeError("S3 cold archive boom")), \
             mock.patch.object(_handoff, "upload_result_to_s3"):
            # Must NOT raise even though cold archive failed.
            result = run_and_report(tmp_path, result_bucket="bucket")

        # The run completed successfully despite the archive failure.
        assert result["process_result_status"] == "events"
        # No error event for the cold-archive failure (best-effort).
        assert _captured_client(mock_run_sim).of("error") == []

    def test_cold_archive_prefix_in_result_event(self, tmp_path: Path, events_env):
        """When cold archive succeeds, the result event carries the prefix."""
        from run_anuga import _handoff

        config = {
            "id": 384, "project": 601, "run_id": 1243,
            "control_server": "https://hydrata.com/",
        }
        (tmp_path / "scenario.json").write_text(json.dumps(config))
        outputs = tmp_path / "outputs_601_384_1243"
        outputs.mkdir()
        (outputs / "result_depth_max.tif").write_bytes(b"fake tif")

        mock_run_sim = mock.MagicMock(return_value=None)
        with mock.patch("run_anuga.run.run_sim", mock_run_sim), \
             mock.patch.object(_handoff, "upload_cold_archive") as mock_archive, \
             mock.patch.object(_handoff, "upload_result_to_s3"):
            run_and_report(tmp_path, result_bucket="bucket")

        mock_archive.assert_called_once()
        fields = _result_event_fields(mock_run_sim)
        assert fields[COLD_ARCHIVE_PREFIX_FIELD] == "cold-archive/601_384_1243/"

    def test_cold_archive_prefix_omitted_when_archive_fails(self, tmp_path: Path, events_env):
        """When cold archive fails the field is OMITTED — never a fabricated
        prefix, and (unlike the deleted report_result kwarg) never an explicit
        null either: the events envelope only carries what actually happened."""
        from run_anuga import _handoff

        config = {
            "id": 384, "project": 601, "run_id": 1243,
            "control_server": "https://hydrata.com/",
        }
        (tmp_path / "scenario.json").write_text(json.dumps(config))
        outputs = tmp_path / "outputs_601_384_1243"
        outputs.mkdir()
        (outputs / "result_depth_max.tif").write_bytes(b"fake tif")

        mock_run_sim = mock.MagicMock(return_value=None)
        with mock.patch("run_anuga.run.run_sim", mock_run_sim), \
             mock.patch.object(_handoff, "upload_cold_archive",
                               side_effect=RuntimeError("boom")), \
             mock.patch.object(_handoff, "upload_result_to_s3"):
            run_and_report(tmp_path, result_bucket="bucket")

        assert COLD_ARCHIVE_PREFIX_FIELD not in _result_event_fields(mock_run_sim)


class TestPlaybackStorePrefixInRunAndReport:
    """TASK-2623 (W1.2, epic 2618) — run_and_report reads the marker
    export_playback_store() left in the output dir (run_sim runs several
    frames above where the exporter actually executes — see
    playback_store.playback_store_prefix_for_run's docstring) and threads
    it into the terminal result event, mirroring cold_archive_prefix exactly."""

    def _base_setup(self, tmp_path, monkeypatch):
        config = {
            "id": 384, "project": 601, "run_id": 1243,
            "control_server": "https://hydrata.com/",
        }
        (tmp_path / "scenario.json").write_text(json.dumps(config))
        outputs = tmp_path / "outputs_601_384_1243"
        outputs.mkdir()
        (outputs / "result_depth_max.tif").write_bytes(b"fake tif")
        return outputs

    def test_playback_store_prefix_posted_when_marker_present(
            self, tmp_path: Path, monkeypatch, events_env):
        outputs = self._base_setup(tmp_path, monkeypatch)
        (outputs / "playback_store_export_result.json").write_text(
            json.dumps({"status": "ok", "s3_prefix": "playback/601_384_1243/", "s3_bucket": "bucket"})
        )
        from run_anuga import _handoff

        mock_run_sim = mock.MagicMock(return_value=None)
        with mock.patch("run_anuga.run.run_sim", mock_run_sim), \
             mock.patch.object(_handoff, "upload_cold_archive"), \
             mock.patch.object(_handoff, "upload_result_to_s3"):
            run_and_report(tmp_path, result_bucket="bucket")

        fields = _result_event_fields(mock_run_sim)
        assert fields[PLAYBACK_STORE_PREFIX_FIELD] == "playback/601_384_1243/"

    def test_playback_store_prefix_omitted_when_marker_absent(
            self, tmp_path: Path, monkeypatch, events_env):
        """No marker (e.g. zarr wasn't installed / export skipped) -> the field
        is omitted, never a stale/fabricated prefix."""
        self._base_setup(tmp_path, monkeypatch)
        from run_anuga import _handoff

        mock_run_sim = mock.MagicMock(return_value=None)
        with mock.patch("run_anuga.run.run_sim", mock_run_sim), \
             mock.patch.object(_handoff, "upload_cold_archive"), \
             mock.patch.object(_handoff, "upload_result_to_s3"):
            run_and_report(tmp_path, result_bucket="bucket")

        assert PLAYBACK_STORE_PREFIX_FIELD not in _result_event_fields(mock_run_sim)

    def test_playback_store_prefix_omitted_when_marker_status_not_ok(
            self, tmp_path: Path, monkeypatch, events_env):
        outputs = self._base_setup(tmp_path, monkeypatch)
        (outputs / "playback_store_export_result.json").write_text(
            json.dumps({"status": "skipped_no_zarr"})
        )
        from run_anuga import _handoff

        mock_run_sim = mock.MagicMock(return_value=None)
        with mock.patch("run_anuga.run.run_sim", mock_run_sim), \
             mock.patch.object(_handoff, "upload_cold_archive"), \
             mock.patch.object(_handoff, "upload_result_to_s3"):
            run_and_report(tmp_path, result_bucket="bucket")

        assert PLAYBACK_STORE_PREFIX_FIELD not in _result_event_fields(mock_run_sim)




# ---------------------------------------------------------------------------
# TASK-2033 — Hardened S3 client factory regression tests
#
# Root cause: prod run-1266 FAILED with S3UploadFailedError — IncompleteBody
# on UploadPart for the slim results zip.  botocore's default
# request_checksum_calculation='when_supported' emits an aws-chunked multipart
# UploadPart whose Content-Length framing can diverge from the declared header.
# Fix: a shared _make_s3_client() factory that sets when_required on BOTH
# upload paths (upload_result_to_s3 + upload_cold_archive).
# ---------------------------------------------------------------------------

class TestHardenedS3ClientFactory:
    """TASK-2033 regression guard: _make_s3_client + both upload functions."""

    def test_make_s3_client_request_checksum_calculation(self):
        """Factory returns a client with request_checksum_calculation='when_required'.

        This assertion FAILS against the pre-fix bare boto3.client("s3") which
        yields client.meta.config.request_checksum_calculation == 'when_supported'
        (botocore default since ~2024).  A failure here means the hardened factory
        is absent or misconfigured — the IncompleteBody regression is back.

        Uses real boto3 (from the hydrata venv) so client.meta.config reflects
        the actual botocore Config the factory supplies; no AWS credentials are
        needed to create the client object.
        """
        from run_anuga import _handoff

        # boto3/botocore are installed in the hydrata venv, not the system Python.
        _VENV = "/opt/venv/hydrata/lib/python3.12/site-packages"
        if _VENV not in sys.path:
            sys.path.insert(0, _VENV)
        boto3_real = pytest.importorskip(
            "boto3", reason="boto3 not available — skipping real-client config check"
        )

        # Call the real factory with real boto3; import_optional is mocked so it
        # returns the real boto3 module rather than raising (boto3 is not in the
        # system Python that runs these tests, only in the venv).
        with mock.patch.object(_handoff, "import_optional", return_value=boto3_real):
            client = _handoff._make_s3_client()

        # client.meta.config reflects exactly what was passed to boto3.client(config=).
        # 'when_required' == checksum only where S3 mandates it (no aws-chunked trailer).
        assert client.meta.config.request_checksum_calculation == "when_required", (
            f"Expected 'when_required', got "
            f"{getattr(client.meta.config, 'request_checksum_calculation', 'ATTR MISSING')!r}. "
            "TASK-2033 regression: _make_s3_client must disable aws-chunked trailer "
            "that caused IncompleteBody on UploadPart in prod run-1266."
        )

    def test_upload_result_to_s3_routes_through_factory(self, tmp_path: Path):
        """upload_result_to_s3 must call _make_s3_client() — not bare boto3.client('s3').

        A future regression that reintroduces a bare boto3.client("s3") in the
        upload path would bypass the hardened Config and re-expose the IncompleteBody
        failure from TASK-2033 prod run-1266.
        """
        from run_anuga import _handoff

        zip_path = tmp_path / "601_384_1266_results.zip"
        zip_path.write_bytes(b"slim results zip")
        mock_s3 = mock.MagicMock()

        with mock.patch.object(_handoff, "_make_s3_client", return_value=mock_s3) as mock_factory, \
             mock.patch.object(_handoff, "import_optional"):
            _handoff.upload_result_to_s3(zip_path, "test-bucket", "601_384_1266_results.zip")

        # The factory MUST have been called (not bypassed by a bare boto3.client call).
        mock_factory.assert_called_once()
        # upload_file must be invoked on the client the factory returned.
        mock_s3.upload_file.assert_called_once()
        call_args = mock_s3.upload_file.call_args
        assert call_args.args[:3] == (
            str(zip_path), "test-bucket", "601_384_1266_results.zip"
        )

    def test_upload_cold_archive_routes_through_factory(self, tmp_path: Path):
        """upload_cold_archive must call _make_s3_client() — not bare boto3.client('s3').

        The cold-archive path shares the same IncompleteBody risk: run-1266's 944 MB
        .sww succeeded via the same multipart path that failed for the slim zip,
        but both paths are hardened because the failure mode is payload-size-sensitive
        and the cost of hardening is zero (TASK-2033).
        """
        from run_anuga import _handoff

        run_label = "601_384_1266"
        output_dir = tmp_path / f"outputs_{run_label}"
        output_dir.mkdir()
        (output_dir / f"{run_label}.sww").write_bytes(b"sww bytes")
        (tmp_path / "package.zip").write_bytes(b"input zip")
        (tmp_path / "scenario.json").write_text("{}")
        mock_s3 = mock.MagicMock()

        with mock.patch.object(_handoff, "_make_s3_client", return_value=mock_s3) as mock_factory, \
             mock.patch.object(_handoff, "import_optional"):
            _handoff.upload_cold_archive(
                tmp_path,
                "test-bucket",
                "cold-archive/601_384_1266/",
                project_id=601,
                scenario_id=384,
                run_id=1266,
            )

        # The factory MUST have been called exactly once (one client, many uploads).
        mock_factory.assert_called_once()
        # upload_file must have been called (at least the .sww + package.zip + scenario.json).
        assert mock_s3.upload_file.call_count >= 3, (
            f"Expected ≥3 upload_file calls, got {mock_s3.upload_file.call_count}"
        )


class TestRunAndReportCallbackWiring:
    """TASK-2663 (epic 2662 W0.1) — regression lock on the dead-progress root cause.

    Since TASK-1924 (W3) run_and_report always passed a truthy
    _EarlyPartialCallback wrapper to run_sim, so run_sim's token-gated
    ``HydrataCallback`` auto-construction (guarded on ``callback is None``)
    never fired and the wrapper delegated every on_progress/on_status to
    inner=None — a silent drop.

    TASK-2681 (W4.1) retired that whole second construction path along with
    the /log/ + /progress/ routes it POSTed to, so the class of bug is now
    structurally impossible rather than guarded. The pin moves with it: with a
    Process uuid present, the wrapper's inner MUST be a live TelemetryCallback
    — never None.
    """

    @pytest.fixture
    def package(self, tmp_path: Path) -> Path:
        config = {
            "id": 384,
            "project": 601,
            "run_id": 1243,
            "control_server": "https://hydrata.com/",
        }
        (tmp_path / "scenario.json").write_text(json.dumps(config))
        outputs = tmp_path / "outputs_601_384_1243"
        outputs.mkdir()
        (outputs / "result_depth_max.tif").write_bytes(b"fake tif")
        return tmp_path

    def _run_and_capture_callback(self, package: Path, telemetry_client=None):
        from run_anuga import _handoff

        mock_run_sim = mock.MagicMock(return_value=None)
        stack = [
            mock.patch("run_anuga.run.run_sim", mock_run_sim),
            mock.patch.object(_handoff, "upload_cold_archive"),
            mock.patch.object(_handoff, "upload_result_to_s3"),
        ]
        # gn_anuga.batch_common is NOT importable from the run_anuga tree, so
        # the real construction site would RAISE here (TASK-2692 fail-closed)
        # for reasons unrelated to what this class pins. Inject the client
        # instead — the assertion under test is what run_and_report WRAPS, not
        # how the client is built (that is TestMakeTelemetryClient's job).
        stack.append(mock.patch.object(
            _handoff, "_make_telemetry_client",
            return_value=telemetry_client,
        ))
        with contextlib.ExitStack() as es:
            for ctx in stack:
                es.enter_context(ctx)
            run_and_report(package, result_bucket="bucket")
        mock_run_sim.assert_called_once()
        return mock_run_sim.call_args.kwargs["callback"]

    def test_process_id_wires_telemetry_callback_inside_wrapper(self, package: Path, monkeypatch):
        """With a Process uuid and callback=None, run_sim must receive a wrapper
        whose inner is a live TelemetryCallback — NOT inner=None (the pre-fix
        silent-drop state this class exists to lock out)."""
        from run_anuga.callbacks import TelemetryCallback

        monkeypatch.setenv("HYDRATA_INTERNAL_COMPUTE_TOKEN", "test-token")
        client = mock.MagicMock(name="TelemetryClient")
        cb = self._run_and_capture_callback(package, telemetry_client=client)
        assert cb is not None
        inner = getattr(cb, "_inner", "MISSING")
        assert isinstance(inner, TelemetryCallback), (
            f"run_sim's callback wrapper carries inner={inner!r}; the TASK-2663 "
            "regression (wrapper around None -> every on_progress dropped) is back"
        )
        assert inner.client is client

    def test_explicit_callback_is_not_replaced(self, package: Path, monkeypatch):
        """An explicitly passed callback (e.g. LoggingCallback on the CLI
        --log-to-stdout path) must ride through unreplaced."""
        from run_anuga import _handoff
        from run_anuga.callbacks import LoggingCallback

        monkeypatch.setenv("HYDRATA_INTERNAL_COMPUTE_TOKEN", "test-token")
        explicit = LoggingCallback()
        mock_run_sim = mock.MagicMock(return_value=None)
        with mock.patch("run_anuga.run.run_sim", mock_run_sim), \
             mock.patch.object(_handoff, "upload_cold_archive"), \
             mock.patch.object(_handoff, "upload_result_to_s3"), \
             mock.patch.object(_handoff, "_make_telemetry_client",
                               return_value=None), \
             mock.patch.dict("os.environ", {ALLOW_UNREPORTED_ENV: "1"}):
            run_and_report(package, callback=explicit, result_bucket="bucket")
        cb = mock_run_sim.call_args.kwargs["callback"]
        assert cb._inner is explicit
