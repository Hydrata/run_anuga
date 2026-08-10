"""Tests for run_anuga.playback_store (TASK-2622, W1.1, epic 2618).

Schema conformance is checked against the signed doc
(docs/reports/2026-08-04-task-2619-playback-store-schema.html, v1) via the
companion validate_playback_store tool — see TestExportAgainstFixtureSww.

zarr-dependent tests are skipped (not failed) when zarr isn't installed,
mirroring the exporter's own import-guard degrade behaviour.
"""
import math
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from run_anuga import playback_store as ps

try:
    import zarr  # noqa: F401

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

requires_zarr = pytest.mark.skipif(not HAS_ZARR, reason="zarr not installed")


# ---------------------------------------------------------------------------
# Quantization math (pure functions, no zarr needed)
# ---------------------------------------------------------------------------

class TestQuantizeRange:
    def test_normal_range(self):
        scale, offset = ps.quantize_range(0.0, 10.0)
        assert offset == 0.0
        assert abs(scale - 10.0 / 65535.0) < 1e-12

    def test_degenerate_range_mandatory_guard(self):
        """schema §3: max==min divides by zero unless guarded — the SWW's own
        friction_range is measured [0.0400, 0.0400], and any fully-dry run
        gives depth [0, 0]. scale must fall back to 1.0, not raise/NaN."""
        scale, offset = ps.quantize_range(0.04, 0.04)
        assert scale == 1.0
        assert offset == 0.04

    def test_fully_dry_run_zero_range(self):
        scale, offset = ps.quantize_range(0.0, 0.0)
        assert scale == 1.0
        assert offset == 0.0

    def test_non_finite_raises(self):
        with pytest.raises(ValueError):
            ps.quantize_range(float("nan"), 1.0)
        with pytest.raises(ValueError):
            ps.quantize_range(0.0, float("inf"))

    def test_inverted_range_raises(self):
        """schema §3: 'a crash-truncated SWW yields a silently negative
        scale' via sww.py's inverted-init range attrs — must be caught."""
        with pytest.raises(ValueError):
            ps.quantize_range(10.0, 0.0)


class TestSymmetricVelocityRange:
    def test_symmetric_and_zero_representable(self):
        """B3: zero must be exactly on the quantization grid."""
        scale, offset = ps.symmetric_velocity_range(5.0)
        assert offset == -5.0
        zero_code = round((0.0 - offset) / scale)
        assert zero_code == 32767

    def test_all_still_water_degenerate(self):
        scale, offset = ps.symmetric_velocity_range(0.0)
        assert scale == 1.0
        assert offset == 0.0

    def test_negative_absmax_raises(self):
        with pytest.raises(ValueError):
            ps.symmetric_velocity_range(-1.0)


class TestQuantizeRoundTrip:
    def test_roundtrip_within_error_bound(self):
        """schema §8: error bound is max_depth/131070."""
        max_depth = 12.5
        scale, offset = ps.quantize_range(0.0, max_depth)
        raw = np.linspace(0, max_depth, 1000, dtype=np.float32)
        q = ps.quantize(raw, scale, offset)
        recon = offset + q.astype(np.float32) * scale
        err_bound = max_depth / 131070.0
        assert np.max(np.abs(recon - raw)) <= err_bound * 1.01  # 1% rounding slack

    def test_clips_out_of_range(self):
        scale, offset = ps.quantize_range(0.0, 10.0)
        raw = np.array([-5.0, 20.0], dtype=np.float32)
        q = ps.quantize(raw, scale, offset)
        assert q[0] == 0
        assert q[1] == 65535


# ---------------------------------------------------------------------------
# Geometry / physics
# ---------------------------------------------------------------------------

class TestComputeInradius:
    def test_equilateral_triangle(self):
        """Equilateral triangle side s: inradius = s / (2*sqrt(3))."""
        side = 2.0
        x = np.array([0.0, side, side / 2], dtype=np.float32)
        y = np.array([0.0, 0.0, side * math.sqrt(3) / 2], dtype=np.float32)
        volumes = np.array([[0, 1, 2]], dtype=np.int32)
        r = ps.compute_inradius(x, y, volumes)
        expected = side / (2 * math.sqrt(3))
        assert abs(r[0] - expected) < 1e-5

    def test_not_area_over_semiperimeter_on_skewed_triangle(self):
        """schema §2: a 3-4-5 right triangle — inradius formula (min
        centroid-to-edge-midpoint distance) must differ from the
        area/semiperimeter formula (0.8333 vs 1.0 — the latter is wrong
        here per the schema's explicit callout)."""
        x = np.array([0.0, 3.0, 0.0], dtype=np.float32)
        y = np.array([0.0, 0.0, 4.0], dtype=np.float32)
        volumes = np.array([[0, 1, 2]], dtype=np.int32)
        r = ps.compute_inradius(x, y, volumes)
        area_over_semiperimeter = 1.0  # area=6, s=6 -> r=1.0 (the WRONG formula here)
        assert abs(r[0] - area_over_semiperimeter) > 0.05

    def test_dtype_and_per_face_shape(self):
        x = np.array([0.0, 1.0, 0.0, 2.0], dtype=np.float32)
        y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        volumes = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        r = ps.compute_inradius(x, y, volumes)
        assert r.dtype == np.float32
        assert r.shape == (2,)  # per-FACE, not per-node


class TestComputeVelocity:
    def test_dry_cell_forced_zero(self):
        depth = np.array([0.0, 1.0])
        momentum = np.array([0.0, 2.0])
        v = ps.compute_velocity(momentum, depth, h0=1e-6)
        assert v[0] == 0.0

    def test_wet_cell_matches_solver_epsilon_formula(self):
        depth = np.array([1.0])
        momentum = np.array([2.0])
        h0 = 1e-6
        v = ps.compute_velocity(momentum, depth, h0)
        expected = 2.0 / (1.0 + h0 / 1.0)
        assert abs(v[0] - expected) < 1e-9

    def test_zero_momentum_and_zero_depth_no_nan(self):
        """A 0/0-shaped input (dry cell, zero momentum) must not produce NaN."""
        depth = np.array([0.0])
        momentum = np.array([0.0])
        v = ps.compute_velocity(momentum, depth, h0=1e-6)
        assert not np.isnan(v[0])
        assert v[0] == 0.0


# ---------------------------------------------------------------------------
# Import-guard / missing-zarr degradation (AC: "missing-zarr degradation
# test green")
# ---------------------------------------------------------------------------

class TestMissingZarrDegradation:
    def test_zarr_available_false_when_import_fails(self):
        with mock.patch.dict("sys.modules", {"zarr": None}):
            assert ps.zarr_available() is False

    def test_export_playback_store_degrades_without_raising(self, tmp_path):
        """The run itself must never fail because zarr is absent."""
        with mock.patch.object(ps, "zarr_available", return_value=False):
            result = ps.export_playback_store(
                input_data={
                    "run_label": "run_1_1_1",
                    "scenario_config": {"project": 1, "id": 1, "run_id": 1, "epsg": "EPSG:28355"},
                },
                sww_path="/nonexistent/does_not_matter.sww",
                output_dir=str(tmp_path),
                upload=False,
            )
        assert result == {"status": "skipped_no_zarr"}

    def test_export_playback_store_degrades_on_any_export_error(self, tmp_path):
        """Any unexpected export failure (bad SWW, disk error, etc.) must
        degrade to status='error', never propagate and fail the run."""
        with mock.patch.object(ps, "zarr_available", return_value=True), \
             mock.patch.object(ps, "_export_playback_store_impl", side_effect=RuntimeError("boom")):
            result = ps.export_playback_store(
                input_data={"run_label": "x", "scenario_config": {}},
                sww_path="whatever.sww",
                output_dir=str(tmp_path),
                upload=False,
            )
        assert result == {"status": "error"}


# ---------------------------------------------------------------------------
# Full export against the session-scoped fixture SWW + schema-conformance via
# the validator tool (AC: "store validates against the W0 schema doc").
# ---------------------------------------------------------------------------

@requires_zarr
@pytest.mark.requires_anuga
class TestExportAgainstFixtureSww:
    # requires_anuga because the session fixture_sww runs a real small sim;
    # the plain CI legs deselect the marker, the e2e job runs it.

    @pytest.fixture(autouse=True)
    def _sww(self, fixture_sww):
        self.sww_path = str(fixture_sww)

    def _export(self, tmp_path, domain=None):
        input_data = {
            "run_label": "run_601_384_1243",
            "scenario_config": {
                "project": 601, "id": 384, "run_id": 1243,
                "epsg": "EPSG:28355",
                "model_start": "2026-01-01T00:00:00+00:00",
                "flow_algorithm": "DE0",
            },
        }
        return ps.export_playback_store(
            input_data=input_data,
            sww_path=self.sww_path,
            output_dir=str(tmp_path),
            domain=domain,
            upload=False,
        )

    def test_export_ok(self, tmp_path):
        result = self._export(tmp_path)
        assert result["status"] == "ok"
        assert Path(result["local_path"]).is_dir()
        import netCDF4

        with netCDF4.Dataset(self.sww_path) as ds:
            expected_n_time = len(ds.variables["time"])
            expected_n_node = len(ds.variables["x"][:])
        assert result["n_time"] == expected_n_time >= 2
        assert result["n_node"] == expected_n_node >= 3

    def test_exported_store_validates_against_schema(self, tmp_path):
        """AC: 'store validates against the W0 schema doc (write an
        array-level check tool and include it)' — this IS that check,
        exercised against a real (fixture) SWW export."""
        from run_anuga.validate_playback_store import validate_store

        result = self._export(tmp_path)
        violations = validate_store(result["local_path"])
        assert violations == [], f"schema violations: {violations}"

    def test_no_zarr_hidden_files(self, tmp_path):
        """B1: a real v3 store has NO .zattrs/.zgroup anywhere — that was
        the v0 error the signed amendment fixed."""
        result = self._export(tmp_path)
        store_path = Path(result["local_path"])
        legacy_v2_files = list(store_path.rglob(".zattrs")) + list(store_path.rglob(".zgroup"))
        assert legacy_v2_files == [], f"v2-style metadata files found: {legacy_v2_files}"

    def test_georef_from_sww_not_scenario_config(self, tmp_path):
        """B2: xllcorner/yllcorner come from the SWW verbatim, NOT from
        scenario_config (which doesn't carry them). Stamp a NONZERO offset
        onto a copy of the fixture — most real SWWs carry xllcorner=0.0,
        which is exactly what hid the B2 defect in the first place, so a
        zero-offset fixture cannot prove the read path."""
        import shutil as _shutil

        import netCDF4
        import zarr

        sww_copy = tmp_path / "georef_offset.sww"
        _shutil.copy(self.sww_path, sww_copy)
        with netCDF4.Dataset(sww_copy, "a") as ds:
            ds.xllcorner = 321000.0
            ds.yllcorner = 5812000.0
        original_sww = self.sww_path
        self.sww_path = str(sww_copy)
        try:
            result = self._export(tmp_path)
        finally:
            self.sww_path = original_sww
        root = zarr.open_group(result["local_path"], mode="r")
        assert root.attrs["xllcorner"] == 321000.0
        assert root.attrs["yllcorner"] == 5812000.0

    def test_epsg_is_verbatim_from_scenario_config_not_sww(self, tmp_path):
        """schema §5: epsg is verbatim from scenario_config['epsg'] — the
        SWW's own zone/hemisphere attrs are untrustworthy (TASK-2634)."""
        import zarr

        result = self._export(tmp_path)
        root = zarr.open_group(result["local_path"], mode="r")
        assert root.attrs["epsg"] == "EPSG:28355"

    def test_false_easting_northing_never_added_to_coords(self, tmp_path):
        """B2: false_easting/false_northing are informational-only attrs —
        node_x/node_y must be stored as-is from the SWW, untouched."""
        import netCDF4
        import zarr

        result = self._export(tmp_path)
        root = zarr.open_group(result["local_path"], mode="r")
        with netCDF4.Dataset(self.sww_path) as ds:
            sww_x = np.array(ds.variables["x"][:], dtype=np.float32)
        np.testing.assert_array_equal(root["node_x"][:], sww_x)

    def test_upload_false_leaves_no_s3_call(self, tmp_path):
        with mock.patch.object(ps, "_upload_store_to_s3") as mock_upload:
            self._export(tmp_path)
        mock_upload.assert_not_called()

    def test_domain_minimum_allowed_height_used_when_domain_passed(self, tmp_path):
        import zarr
        from types import SimpleNamespace

        fake_domain = SimpleNamespace(minimum_allowed_height=1.23e-12, flow_algorithm="DE1")
        result = self._export(tmp_path, domain=fake_domain)
        root = zarr.open_group(result["local_path"], mode="r")
        assert root.attrs["minimum_allowed_height"] == 1.23e-12
        assert root.attrs["flow_algorithm"] == "DE1"


class TestUploadWritesCacheControl:
    """TASK-2709 (W2.1, epic 2706) — every uploaded playback object must carry
    a Cache-Control header, written at EXPORT (S3 has no way to add one later
    without rewriting the object).

    Why it matters: the manifest's chunk URLs are the ONLY way the browser
    fetches the store, and without a cache directive the browser revalidates
    (or simply re-downloads) all 62.7 MiB of geometry on every single page
    load. This pairs with TASK-2710: a stable-within-a-time-bucket URL is what
    gives the browser cache a stable key to hit, and the rotating bucket is
    what bounds how long a stale object can be served despite max-age=1y.
    """

    def _upload_with_fake_client(self, tmp_path):
        """Injects fake boto3 modules via sys.modules rather than patching
        ``boto3.client``: boto3 is NOT installed under /usr/bin/python, which
        is the interpreter the documented `python -m pytest tests -k playback`
        command uses. Patching a module that cannot be imported would make this
        test silently env-dependent."""
        import sys

        store = tmp_path / "store"
        (store / "depth" / "c" / "0").mkdir(parents=True)
        (store / "zarr.json").write_text("{}")
        (store / "depth" / "c" / "0" / "0").write_bytes(b"\x00\x01")

        fake_s3 = mock.MagicMock()
        fake_boto3 = mock.MagicMock()
        fake_boto3.client.return_value = fake_s3
        fake_modules = {
            "boto3": fake_boto3,
            "boto3.s3": mock.MagicMock(),
            "boto3.s3.transfer": mock.MagicMock(),
        }
        with mock.patch.dict(sys.modules, fake_modules):
            ps._upload_store_to_s3(store, "test-bucket", "playback/1_1_1/")
        return fake_s3

    def test_every_object_is_uploaded_with_cache_control(self, tmp_path):
        fake_s3 = self._upload_with_fake_client(tmp_path)

        assert fake_s3.upload_file.call_count == 2, "both store files must upload"
        for call in fake_s3.upload_file.call_args_list:
            extra_args = call.kwargs.get("ExtraArgs")
            assert extra_args is not None, (
                f"upload_file({call.args[2]!r}) passed no ExtraArgs, so no "
                f"cache directive is written on the object"
            )
            assert extra_args.get("CacheControl") == ps.PLAYBACK_CACHE_CONTROL

    def test_cache_control_value_is_immutable_and_long_lived(self):
        """Pins the exact directive the AC names. `immutable` is what stops the
        browser issuing a revalidation request per chunk; without it a
        conditional GET per chunk still costs a round-trip each."""
        assert ps.PLAYBACK_CACHE_CONTROL == "public, max-age=31536000, immutable"


@requires_zarr
class TestValidatorCatchesRealDefects:
    """Proves the validator is a real detector, not a rubber stamp —
    feeds it a deliberately-broken store and checks it's caught."""

    def _write_minimal_store(self, tmp_path, *, fill_value=0, omit_attrs=False):
        import zarr
        from zarr.codecs import BytesCodec, GzipCodec

        store_path = tmp_path / "broken.zarr"
        root = zarr.open_group(str(store_path), mode="w", zarr_format=3)
        if not omit_attrs:
            root.attrs["format_version"] = 1
        arr = root.create_array(
            "depth", shape=(3, 10), chunks=(10, 10), dtype="uint16",
            filters=[], serializer=BytesCodec(endian="little"),
            compressors=GzipCodec(level=6), fill_value=fill_value,
            chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        )
        arr[:] = np.zeros((3, 10), dtype="uint16")
        return store_path

    def test_catches_wrong_fill_value(self, tmp_path):
        from run_anuga.validate_playback_store import validate_store

        store_path = self._write_minimal_store(tmp_path, fill_value=999)
        violations = validate_store(store_path)
        assert any("fill_value" in v and "999" in v for v in violations)

    def test_catches_missing_arrays(self, tmp_path):
        from run_anuga.validate_playback_store import validate_store

        store_path = self._write_minimal_store(tmp_path)
        violations = validate_store(store_path)
        assert any("node_x" in v for v in violations)
        assert any("x_velocity" in v for v in violations)

    def test_catches_missing_group_attrs(self, tmp_path):
        from run_anuga.validate_playback_store import validate_store

        store_path = self._write_minimal_store(tmp_path, omit_attrs=True)
        violations = validate_store(store_path)
        assert any("xllcorner" in v for v in violations)

    @pytest.mark.requires_anuga
    def test_valid_store_zero_violations(self, tmp_path, fixture_sww):
        """Negative control: the exporter's own real output must be clean
        (guards against the validator being too strict, not just too loose)."""
        from run_anuga.validate_playback_store import validate_store

        input_data = {
            "run_label": "run_1_1_1",
            "scenario_config": {"project": 1, "id": 1, "run_id": 1, "epsg": "EPSG:28355"},
        }
        result = ps.export_playback_store(
            input_data=input_data, sww_path=str(fixture_sww),
            output_dir=str(tmp_path), upload=False,
        )
        assert validate_store(result["local_path"]) == []


class TestMakePlaybackStorePrefix:
    def test_shape_matches_cold_archive_convention(self):
        assert ps.make_playback_store_prefix(601, 384, 1243) == "playback/601_384_1243/"


class TestPlaybackStoreMarker:
    """TASK-2623 (W1.2, epic 2618) — the cross-process handoff channel
    _handoff.run_and_report() uses to learn the uploaded S3 prefix, since
    export_playback_store() runs several frames deeper (inside run_sim())
    than where report_result() is posted."""

    def test_marker_written_on_skip(self, tmp_path):
        with mock.patch.object(ps, "zarr_available", return_value=False):
            ps.export_playback_store(
                input_data={"run_label": "x", "scenario_config": {}},
                sww_path="whatever.sww", output_dir=str(tmp_path), upload=False,
            )
        marker = ps.read_playback_store_marker(tmp_path)
        assert marker == {"status": "skipped_no_zarr"}

    def test_marker_written_on_error(self, tmp_path):
        with mock.patch.object(ps, "zarr_available", return_value=True), \
             mock.patch.object(ps, "_export_playback_store_impl", side_effect=RuntimeError("boom")):
            ps.export_playback_store(
                input_data={"run_label": "x", "scenario_config": {}},
                sww_path="whatever.sww", output_dir=str(tmp_path), upload=False,
            )
        marker = ps.read_playback_store_marker(tmp_path)
        assert marker == {"status": "error"}

    def test_marker_absent_returns_none(self, tmp_path):
        assert ps.read_playback_store_marker(tmp_path) is None

    def test_playback_store_prefix_for_run_none_when_no_marker(self, tmp_path):
        assert ps.playback_store_prefix_for_run(tmp_path, 601, 384, 1243) is None

    def test_playback_store_prefix_for_run_none_when_skipped(self, tmp_path):
        output_dir = tmp_path / "outputs_601_384_1243"
        output_dir.mkdir()
        with mock.patch.object(ps, "zarr_available", return_value=False):
            ps.export_playback_store(
                input_data={"run_label": "x", "scenario_config": {}},
                sww_path="whatever.sww", output_dir=str(output_dir), upload=False,
            )
        assert ps.playback_store_prefix_for_run(tmp_path, 601, 384, 1243) is None

    @requires_zarr
    @pytest.mark.requires_anuga
    def test_playback_store_prefix_for_run_returns_prefix_after_real_export(self, tmp_path, fixture_sww):
        output_dir = tmp_path / "outputs_601_384_1243"
        output_dir.mkdir()
        with mock.patch.object(ps, "_upload_store_to_s3") as mock_upload:
            mock_upload.return_value = "playback/601_384_1243/"
            ps.export_playback_store(
                input_data={
                    "run_label": "run_601_384_1243",
                    "scenario_config": {"project": 601, "id": 384, "run_id": 1243, "epsg": "EPSG:28355"},
                },
                sww_path=str(fixture_sww), output_dir=str(output_dir),
                upload=True, bucket="test-bucket",
            )
        prefix = ps.playback_store_prefix_for_run(tmp_path, 601, 384, 1243)
        assert prefix == "playback/601_384_1243/"


# Module-level conditional mark, same style as requires_zarr above (and
# counted once by the marker taxonomy the same way): opt-in only (needs real
# AWS creds + network) — set RUN_ANUGA_LIVE_S3_PLAYBACK_TEST=1 to run. NEVER
# targets anuga-result-storage and never deletes anything; see TASK-2622 W1
# wave-agent verification, 2026-08-06, for a manual run that uploaded 20
# objects to a playback/W1-WAVE-AGENT-VERIFICATION-2026-08-06-2622/ test
# prefix on anuga-test-storage.
live_s3_opt_in = pytest.mark.skipif(
    not os.environ.get("RUN_ANUGA_LIVE_S3_PLAYBACK_TEST"),
    reason="opt-in only — set RUN_ANUGA_LIVE_S3_PLAYBACK_TEST=1 (TASK-2622)",
)


@live_s3_opt_in
class TestLiveS3CacheControl:
    """TASK-2709 (W2.1, epic 2706) — the Cache-Control directive must survive
    the REAL boto3 upload, checked with a real ``head_object``.

    Deliberately NOT gated on zarr/ANUGA (unlike TestLiveS3Upload below): the
    thing under test is ``_upload_store_to_s3``'s ExtraArgs plumbing, which
    uploads whatever files it is given and has no zarr dependency at all. A
    directory of bytes is a faithful stand-in for a store here, and dropping
    the gate is what lets this proof actually run on a box where zarr is
    absent — which is every interpreter on the 2026-08-10 workstation.
    """

    def test_cache_control_lands_on_real_s3_objects(self, tmp_path):
        import time

        import boto3

        bucket = "anuga-test-storage"  # NEVER anuga-result-storage from a test
        prefix = f"playback/pytest-cachecontrol-{int(time.time())}/"

        store = tmp_path / "store"
        (store / "depth" / "c" / "0").mkdir(parents=True)
        (store / "zarr.json").write_text('{"zarr_format": 3}')
        # >8 MiB so upload_file takes the MULTIPART branch (the exporter's
        # TransferConfig multipart_threshold): ExtraArgs is easiest to lose
        # there, because multipart carries it on create_multipart_upload
        # rather than on the PUT.
        (store / "depth" / "c" / "0" / "0").write_bytes(b"\x00" * (9 * 1024 * 1024))

        ps._upload_store_to_s3(store, bucket, prefix)

        s3 = boto3.client("s3")
        keys = [
            o["Key"]
            for o in s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
        ]
        assert len(keys) == 2, keys
        for key in keys:
            head = s3.head_object(Bucket=bucket, Key=key)
            assert head.get("CacheControl") == ps.PLAYBACK_CACHE_CONTROL, key
        # Never deletes — wave brief hard rule.


@requires_zarr
@live_s3_opt_in
@pytest.mark.requires_anuga
class TestLiveS3Upload:
    """Real S3 round-trip proof — the AC-sanctioned fallback verification
    path when a full localhost ANUGA run is impractical inside a wave:
    'prove exporter + upload against a fixture/real SWW with upload to a
    unique test prefix on anuga-test-storage only'. Opt-in because it needs
    real AWS credentials; the manual verification already performed for
    TASK-2622 stands as the AC proof regardless of whether CI ever opts in.
    """

    def test_uploads_and_lists_back_from_s3(self, tmp_path, fixture_sww):
        import time

        import boto3

        bucket = "anuga-test-storage"  # NEVER anuga-result-storage from a test
        prefix = f"playback/pytest-live-verify-{int(time.time())}/"
        input_data = {
            "run_label": "run_1_1_1",
            "scenario_config": {"project": 1, "id": 1, "run_id": 1, "epsg": "EPSG:28355"},
        }
        result = ps.export_playback_store(
            input_data=input_data, sww_path=str(fixture_sww),
            output_dir=str(tmp_path), upload=True, bucket=bucket, prefix=prefix,
        )
        assert result["status"] == "ok"
        assert result["s3_bucket"] == bucket

        s3 = boto3.client("s3")
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        keys = [o["Key"] for o in resp.get("Contents", [])]
        assert any(k.endswith("zarr.json") for k in keys)

        # TASK-2709 (W2.1, epic 2706) — the cache directive must survive the
        # REAL boto3 upload path, not just a mocked call assertion: ExtraArgs
        # is exactly the kind of argument that a TransferConfig/multipart path
        # can drop, and a mocked verify step is an unverified step.
        for key in keys:
            head = s3.head_object(Bucket=bucket, Key=key)
            assert head.get("CacheControl") == ps.PLAYBACK_CACHE_CONTROL, key

        # Deliberately does NOT delete the uploaded objects — never delete
        # any S3 object from a test (wave brief hard rule).
