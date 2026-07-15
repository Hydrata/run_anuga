"""Tests for TASK-1921 — reprocess_from_archived_sww entrypoint smoke.

De-risks the write-only-archive bet (TASK-1921 / W2): proves the seam that a
future flow-animation build can reopen a cold-archived .sww and re-render a
quantity.

Strategy
--------
* ``reprocess_from_archived_sww`` pulls a .sww from S3 cold archive (boto3)
  then calls ``Make_Geotif`` (anuga) to produce a max raster.
* Both are optional-dep: boto3 via ``import_optional``, anuga via the same.
* In CI (no anuga, no boto3 on the test path) we mock BOTH heavy deps:
  - s3.download_file is patched to write a placeholder .sww into the temp dir.
  - Make_Geotif is patched to write a 1-byte sentinel .tif alongside the .sww.
  This validates the integration plumbing without requiring the full sim stack.
* The mock pattern mirrors test_handoff.py's existing approach (MagicMock + patch).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Helper: detect whether the heavy deps are actually available so we can
# document whether the "real" or "mocked" path ran.
# ---------------------------------------------------------------------------
_HAS_BOTO3 = importlib.util.find_spec("boto3") is not None
_HAS_ANUGA = importlib.util.find_spec("anuga") is not None


class TestReprocessFromArchivedSww:
    """Smoke tests for reprocess_from_archived_sww in run_utils.py (TASK-1921)."""

    def test_reprocess_is_importable(self):
        """reprocess_from_archived_sww must be importable without heavy deps."""
        from run_anuga.run_utils import reprocess_from_archived_sww
        assert callable(reprocess_from_archived_sww)

    def test_reprocess_mocked_s3_and_make_geotif(self, tmp_path: Path):
        """Integration smoke: mocked S3 fetch + mocked Make_Geotif.

        This is the portable proof that runs on any Python install — it
        validates the plumbing (S3 fetch -> local .sww -> Make_Geotif -> raster
        path returned) without requiring anuga or boto3.
        """
        from run_anuga import run_utils

        # The .sww key in the cold archive.
        bucket = "test-result-bucket"
        sww_key = "cold-archive/601_384_1243/601_384_1243.sww"

        # Sentinel raster that Make_Geotif "produces".
        expected_raster_name = "601_384_1243_depth_max.tif"
        expected_raster_content = b"fake raster content"

        def _fake_download_file(bucket_arg, key_arg, local_path_arg):
            """Simulate s3.download_file by writing a placeholder .sww."""
            Path(local_path_arg).write_bytes(b"fake sww bytes")

        def _fake_make_geotif(**kwargs):
            """Simulate Make_Geotif by writing a sentinel *_max.tif."""
            out_dir = Path(kwargs.get("output_dir", str(tmp_path)))
            raster = out_dir / expected_raster_name
            raster.write_bytes(expected_raster_content)

        # Patch import_optional to return controlled mocks.
        fake_boto3 = mock.MagicMock()
        fake_boto3.client.return_value.download_file.side_effect = _fake_download_file

        fake_util = mock.MagicMock()
        fake_util.Make_Geotif.side_effect = _fake_make_geotif
        fake_anuga = mock.MagicMock()
        fake_anuga.utilities.plot_utils = fake_util

        def _import_optional_side_effect(name):
            if name == "boto3":
                return fake_boto3
            if name == "anuga":
                return fake_anuga
            raise ImportError(f"import_optional({name!r}) unexpected in this test")

        with mock.patch.object(run_utils, "import_optional", side_effect=_import_optional_side_effect):
            result = run_utils.reprocess_from_archived_sww(
                bucket,
                sww_key,
                output_dir=str(tmp_path),
                quantity="depth",
                cell_size=10.0,
                epsg_code=32754,
            )

        # The function must return the raster path.
        assert result is not None, "reprocess_from_archived_sww must return a raster path"
        result_path = Path(result)
        assert result_path.exists(), f"Returned raster path {result_path} does not exist"
        assert result_path.stat().st_size > 0, "Returned raster must be non-empty"
        assert result_path.name.endswith("_depth_max.tif"), (
            f"Expected *_depth_max.tif, got {result_path.name!r}"
        )

        # S3 download must have been called with the correct key.
        fake_boto3.client.return_value.download_file.assert_called_once()
        call_args = fake_boto3.client.return_value.download_file.call_args[0]
        assert call_args[0] == bucket, f"Expected bucket {bucket!r}, got {call_args[0]!r}"
        assert call_args[1] == sww_key, f"Expected key {sww_key!r}, got {call_args[1]!r}"

        # Make_Geotif must have been called for the right quantity and time-step.
        fake_util.Make_Geotif.assert_called_once()
        geotif_kwargs = fake_util.Make_Geotif.call_args.kwargs
        assert geotif_kwargs.get("output_quantities") == ["depth"]
        assert geotif_kwargs.get("myTimeStep") == "max"
        assert geotif_kwargs.get("EPSG_CODE") == 32754

    def test_reprocess_passes_cell_size_to_make_geotif(self, tmp_path: Path):
        """cell_size kwarg must reach Make_Geotif as CellSize."""
        from run_anuga import run_utils

        def _fake_download(bucket, key, local_path):
            Path(local_path).write_bytes(b"sww")

        def _fake_make_geotif(**kwargs):
            out_dir = Path(kwargs.get("output_dir", str(tmp_path)))
            (out_dir / "601_384_1243_depth_max.tif").write_bytes(b"tif")

        fake_boto3 = mock.MagicMock()
        fake_boto3.client.return_value.download_file.side_effect = _fake_download
        fake_util = mock.MagicMock()
        fake_util.Make_Geotif.side_effect = _fake_make_geotif
        fake_anuga = mock.MagicMock()
        fake_anuga.utilities.plot_utils = fake_util

        def _import_optional(name):
            return fake_boto3 if name == "boto3" else fake_anuga

        with mock.patch.object(run_utils, "import_optional", side_effect=_import_optional):
            run_utils.reprocess_from_archived_sww(
                "bucket",
                "cold-archive/601_384_1243/601_384_1243.sww",
                output_dir=str(tmp_path),
                cell_size=50.0,
            )

        geotif_kwargs = fake_util.Make_Geotif.call_args.kwargs
        assert geotif_kwargs.get("CellSize") == 50.0

    def test_reprocess_raises_when_raster_not_produced(self, tmp_path: Path):
        """If Make_Geotif produces no raster, FileNotFoundError is raised."""
        from run_anuga import run_utils

        def _fake_download(bucket, key, local_path):
            Path(local_path).write_bytes(b"sww")

        def _fake_make_geotif(**kwargs):
            # Deliberately produce NO raster.
            pass

        fake_boto3 = mock.MagicMock()
        fake_boto3.client.return_value.download_file.side_effect = _fake_download
        fake_util = mock.MagicMock()
        fake_util.Make_Geotif.side_effect = _fake_make_geotif
        fake_anuga = mock.MagicMock()
        fake_anuga.utilities.plot_utils = fake_util

        def _import_optional(name):
            return fake_boto3 if name == "boto3" else fake_anuga

        with mock.patch.object(run_utils, "import_optional", side_effect=_import_optional):
            with pytest.raises(FileNotFoundError, match="Make_Geotif did not produce"):
                run_utils.reprocess_from_archived_sww(
                    "bucket",
                    "cold-archive/601_384_1243/601_384_1243.sww",
                    output_dir=str(tmp_path),
                )

    def test_reprocess_default_quantity_is_depth(self, tmp_path: Path):
        """Default quantity must be 'depth' (cheapest, most useful single frame)."""
        from run_anuga import run_utils

        def _fake_download(bucket, key, local_path):
            Path(local_path).write_bytes(b"sww")

        def _fake_make_geotif(**kwargs):
            out_dir = Path(kwargs.get("output_dir", str(tmp_path)))
            quantity = kwargs.get("output_quantities", ["depth"])[0]
            (out_dir / f"601_384_1243_{quantity}_max.tif").write_bytes(b"tif")

        fake_boto3 = mock.MagicMock()
        fake_boto3.client.return_value.download_file.side_effect = _fake_download
        fake_util = mock.MagicMock()
        fake_util.Make_Geotif.side_effect = _fake_make_geotif
        fake_anuga = mock.MagicMock()
        fake_anuga.utilities.plot_utils = fake_util

        def _import_optional(name):
            return fake_boto3 if name == "boto3" else fake_anuga

        with mock.patch.object(run_utils, "import_optional", side_effect=_import_optional):
            result = run_utils.reprocess_from_archived_sww(
                "bucket",
                "cold-archive/601_384_1243/601_384_1243.sww",
                output_dir=str(tmp_path),
            )

        assert "depth" in result.name

    @pytest.mark.skipif(
        not (_HAS_BOTO3 and _HAS_ANUGA),
        reason="real integration requires boto3+anuga extras; mocked smoke above is the CI proof",
    )
    def test_real_deps_available_marker(self):
        """Marker: when boto3 + anuga are both present, the function is importable."""
        from run_anuga.run_utils import reprocess_from_archived_sww
        assert callable(reprocess_from_archived_sww)
