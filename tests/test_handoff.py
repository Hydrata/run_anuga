"""Tests for run_anuga._handoff — TASK-1159 (F1) result handoff.

Covers:

* The shared field-name constant (``RESULT_PACKAGE_KEY_FIELD``).
* ``make_result_key`` matches the entrypoint.sh shape.
* ``zip_outputs`` respects the entrypoint's exclusion list (``package.zip``,
  the result zip itself, and any embedded ``run_anuga/`` source tree).
* ``upload_result_to_s3`` calls ``boto3.client('s3').upload_file`` and is
  gated by ``import_optional`` so importing the module without ``boto3``
  installed does not raise.
* ``report_result`` POSTs ``{result_package_key: ...}`` to the V2
  ``/process-result/`` endpoint and uses the shared constant.
* ``report_error`` POSTs ``{message, source?}`` to the V2 ``/error/`` endpoint.
* ``run_and_report`` orchestrates run_sim + zip + upload + POST, and
  POSTs ``/error/`` on failure (the F0 wedge defence).
* The module imports cleanly without Django (run_anuga must stay Django-free).
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from run_anuga._handoff import (
    RESULT_PACKAGE_KEY_FIELD,
    make_result_key,
    report_error,
    report_result,
    run_and_report,
    zip_outputs,
)


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
        outputs = self._make_realistic_outputs(tmp_path)
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
        zip_path = tmp_path / "result.zip"
        zip_path.write_bytes(b"zip")
        from run_anuga import _handoff

        with mock.patch.object(_handoff, "import_optional") as mock_import:
            mock_s3 = mock.MagicMock()
            mock_import.return_value.client.return_value = mock_s3
            _handoff.upload_result_to_s3(zip_path, "test-bucket", "601_384_1243_results.zip")
        mock_import.assert_called_once_with("boto3")
        mock_s3.upload_file.assert_called_once_with(
            str(zip_path), "test-bucket", "601_384_1243_results.zip"
        )


class TestReportResult:
    def test_posts_result_package_key_shape(self, monkeypatch):
        monkeypatch.setenv("HYDRATA_INTERNAL_COMPUTE_TOKEN", "test-token")
        session = mock.MagicMock()
        session.headers = {}
        response = mock.MagicMock(status_code=202, text="")
        session.post.return_value = response

        with mock.patch(
            "run_anuga._http.import_optional"
        ) as mock_import:
            mock_import.return_value.Session.return_value = session
            result = report_result(
                "https://hydrata.com/", run_id=99, token="test-token", result_key="1_2_3_results.zip"
            )

        # Receiver reads request.data via the shared constant; sender must send it.
        called_url, called_kwargs = session.post.call_args[0], session.post.call_args[1]
        assert called_url[0].endswith("/api/v2/anuga/runs/99/process-result/")
        assert called_kwargs["data"] == {RESULT_PACKAGE_KEY_FIELD: "1_2_3_results.zip"}
        assert result is response


class TestReportError:
    def test_posts_message_and_source(self, monkeypatch):
        monkeypatch.setenv("HYDRATA_INTERNAL_COMPUTE_TOKEN", "test-token")
        session = mock.MagicMock()
        session.headers = {}
        session.post.return_value = mock.MagicMock(status_code=201, text="")

        with mock.patch("run_anuga._http.import_optional") as mock_import:
            mock_import.return_value.Session.return_value = session
            report_error(
                "https://hydrata.com/",
                run_id=99,
                token="test-token",
                message="sim crashed",
                source="run_and_report",
            )

        called_url, called_kwargs = session.post.call_args[0], session.post.call_args[1]
        assert called_url[0].endswith("/api/v2/anuga/runs/99/error/")
        assert called_kwargs["data"] == {"message": "sim crashed", "source": "run_and_report"}


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

    def test_happy_path_zips_uploads_and_posts(self, package: Path, monkeypatch):
        monkeypatch.setenv("HYDRATA_INTERNAL_COMPUTE_TOKEN", "test-token")
        from run_anuga import _handoff

        mock_run_sim = mock.MagicMock(return_value=None)
        post_response = mock.MagicMock(status_code=202, text="")
        # `run_sim` is imported lazily inside run_and_report (`from run_anuga.run
        # import run_sim`) so patch the source module, not _handoff.
        with mock.patch("run_anuga.run.run_sim", mock_run_sim), \
             mock.patch.object(_handoff, "upload_result_to_s3") as mock_upload, \
             mock.patch.object(_handoff, "report_result", return_value=post_response) as mock_post, \
             mock.patch.object(_handoff, "report_error") as mock_err:
            result = run_and_report(package, result_bucket="bucket")

        mock_run_sim.assert_called_once()
        mock_upload.assert_called_once()
        upload_args = mock_upload.call_args[0]
        assert upload_args[1] == "bucket"
        assert upload_args[2] == "601_384_1243_results.zip"
        mock_post.assert_called_once_with(
            "https://hydrata.com/", 1243, "test-token", "601_384_1243_results.zip"
        )
        # No /error/ POST on the happy path — a future regression that POSTed
        # /error/ on success would otherwise silently slip through.
        mock_err.assert_not_called()
        assert result == {"result_key": "601_384_1243_results.zip", "process_result_status": 202}

    def test_sim_failure_posts_error_then_raises(self, package: Path, monkeypatch):
        monkeypatch.setenv("HYDRATA_INTERNAL_COMPUTE_TOKEN", "test-token")
        from run_anuga import _handoff

        with mock.patch("run_anuga.run.run_sim", side_effect=RuntimeError("sim boom")), \
             mock.patch.object(_handoff, "report_error") as mock_err, \
             mock.patch.object(_handoff, "upload_result_to_s3") as mock_upload:
            with pytest.raises(RuntimeError, match="sim boom"):
                run_and_report(package, result_bucket="bucket")

        mock_upload.assert_not_called()
        mock_err.assert_called_once()
        kwargs = mock_err.call_args.kwargs
        assert kwargs["source"] == "run_and_report"

    def test_non_2xx_process_result_posts_error(self, package: Path, monkeypatch):
        monkeypatch.setenv("HYDRATA_INTERNAL_COMPUTE_TOKEN", "test-token")
        from run_anuga import _handoff

        bad_response = mock.MagicMock(status_code=500, text="boom")
        with mock.patch("run_anuga.run.run_sim", return_value=None), \
             mock.patch.object(_handoff, "upload_result_to_s3"), \
             mock.patch.object(_handoff, "report_result", return_value=bad_response), \
             mock.patch.object(_handoff, "report_error") as mock_err:
            with pytest.raises(RuntimeError, match="HTTP 500"):
                run_and_report(package, result_bucket="bucket")

        mock_err.assert_called_once()
