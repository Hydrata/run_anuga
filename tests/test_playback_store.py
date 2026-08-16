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


class TestDeriveChunkLengthT:
    """TASK-2719 (epic 2706 W8, decision D5) — adaptive time-chunk length,
    a PURE function of n_node alone. No store written here (AC1)."""

    def test_floor_at_run_1328_scale(self):
        """n_node=3,393,075 (run 1328) lands on the FLOOR, never 1 — D5
        forbids chunk length 1 (it recreates the 2618 client-side LRU
        thrash by construction)."""
        result = ps.derive_chunk_length_t(3_393_075)
        assert result == 2, (
            f"expected the D5 floor of 2 for run-1328 scale, got {result} — "
            "D5 forbids chunk length 1"
        )

    def test_cap_for_small_meshes_byte_identical_to_today(self):
        """n_node=50,000 is well under the 838,860 crossover -> the CAP, 10,
        byte-identical to every store exported before this task."""
        assert ps.derive_chunk_length_t(50_000) == 10

    def test_interior_value(self):
        assert ps.derive_chunk_length_t(1_000_000) == 8

    def test_floor_crossover_pinned_exactly(self):
        """The exact boundary, both sides, per the verified clamp ladder."""
        assert ps.derive_chunk_length_t(2_796_202) == 3
        assert ps.derive_chunk_length_t(2_796_203) == 2

    def test_floor_never_goes_below_two_arbitrarily_large_mesh(self):
        assert ps.derive_chunk_length_t(50_000_000) == 2

    def test_pure_function_of_n_node_alone(self):
        """Same n_node, called independently (no n_time argument exists) ->
        same result every time — the function has no other input."""
        assert ps.derive_chunk_length_t(1_300_000) == ps.derive_chunk_length_t(1_300_000)
        assert ps.derive_chunk_length_t(1_300_000) == 6


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
# TASK-2752 (W8.2, epic 2706) — the temporal-max envelope. AC1/AC2.
# ---------------------------------------------------------------------------

class TestComputeEnvelopes:
    """AC2 "THE TRAP IS TESTED": a fixture where x_velocity and y_velocity
    peak at DIFFERENT timesteps must prove ``velocity_max !=
    magnitude(x_velocity_max, y_velocity_max)`` and that the shipped array is
    the former (max-of-derived-speed), not the latter (derived-of-component-
    max).

    Node 0 is the trap: x_velocity peaks at t=0 (vx=3, vy=0 -> speed 3),
    y_velocity peaks at t=1 (vx=0, vy=4 -> speed 4) — the classic 3-4-5
    triangle, chosen so a naive `sqrt(max(vx)**2 + max(vy)**2)` answers
    exactly 5 (combining two different instants' peaks) where the correct
    per-timestep-derived-then-maxed answer is 4 (t=1's real speed).
    Node 1 is a no-trap control (velocity never rotates directions), where
    naive and correct agree — proving this isn't a fixture that happens to
    disagree with everything.

    RED PROOF (recorded, not re-run here): this test was run once against a
    deliberately naive `compute_envelopes` that computed
    `sqrt(max(|x_velocity|)**2 + max(|y_velocity|)**2)` in place of the
    correct max-of-derived-speed; it failed exactly as this test asserts it
    must. See docs/epic-state/wave-reports/TASK-2706-W8.2-evidence/
    2752-ac2-trap-red-proof.txt for the captured pytest output.
    """

    DEPTH = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    X_VEL = np.array([[3.0, 2.0], [0.0, 2.0], [0.0, 2.0]], dtype=np.float32)
    Y_VEL = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 0.0]], dtype=np.float32)

    def test_velocity_max_is_max_of_derived_speed_not_derived_of_component_max(self):
        result = ps.compute_envelopes(self.DEPTH, self.X_VEL, self.Y_VEL)
        naive_component_max = np.sqrt(
            np.max(np.abs(self.X_VEL), axis=0) ** 2 + np.max(np.abs(self.Y_VEL), axis=0) ** 2
        )
        # The fixture itself must actually distinguish the two orders of
        # operation at node 0, or the assertion below proves nothing.
        assert naive_component_max[0] != 4.0, "trap fixture is degenerate — naive and correct coincide"
        assert result["velocity"][0] != naive_component_max[0]
        np.testing.assert_allclose(result["velocity"], [4.0, 2.0])
        np.testing.assert_allclose(naive_component_max, [5.0, 2.0])

    def test_depth_max_is_max_over_time(self):
        result = ps.compute_envelopes(self.DEPTH, self.X_VEL, self.Y_VEL)
        np.testing.assert_allclose(result["depth"], [1.0, 1.0])

    def test_div_max_is_max_of_depth_times_derived_speed_same_instant(self):
        """dIV = depth * speed, derived PER TIMESTEP then maxed — mirrors the
        velocity trap: div at node 0 is [3, 4, 0] (depth 1 * speed at each t),
        max 4, NOT depth_max * velocity_max (which would give 1*4=4 here by
        coincidence, but is the wrong formula in general — dIV_max is its own
        max-of-derived quantity, not a product of two other envelopes)."""
        result = ps.compute_envelopes(self.DEPTH, self.X_VEL, self.Y_VEL)
        np.testing.assert_allclose(result["div"], [4.0, 2.0])

    def test_all_three_envelopes_present_non_negative_float32(self):
        result = ps.compute_envelopes(self.DEPTH, self.X_VEL, self.Y_VEL)
        assert set(result.keys()) == set(ps.ENVELOPE_QUANTITIES)
        for arr in result.values():
            assert arr.dtype == np.float32
            assert np.all(arr >= 0)

    def test_empty_arrays_return_zeros_not_crash(self):
        empty_depth = np.zeros((0, 3), dtype=np.float32)
        empty_v = np.zeros((0, 3), dtype=np.float32)
        result = ps.compute_envelopes(empty_depth, empty_v, empty_v)
        for name in ps.ENVELOPE_QUANTITIES:
            assert result[name].shape == (3,)
            assert np.all(result[name] == 0)


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

    def test_envelope_arrays_present_and_declared(self, tmp_path):
        """AC1/AC4 — the producer writes depth_max/velocity_max/div_max and
        declares them in group attrs, following has_dt's first-class-
        absence shape (a fresh export always declares all three today)."""
        import zarr

        result = self._export(tmp_path)
        root = zarr.open_group(result["local_path"], mode="r")
        assert list(root.attrs["envelope_quantities"]) == list(ps.ENVELOPE_QUANTITIES)
        n_node = result["n_node"]
        for name in ps.ENVELOPE_QUANTITIES:
            arr = root[f"{name}_max"]
            assert arr.shape == (n_node,), f"{name}_max: shape={arr.shape}, expected ({n_node},)"
            assert str(arr.dtype) == "uint16", f"{name}_max: dtype={arr.dtype}"
            for qattr in ("scale", "offset", "quantized_dtype", "byteorder", "valid_min", "valid_max"):
                assert qattr in arr.attrs, f"{name}_max missing quantization attr '{qattr}'"

    def test_depth_max_dequantizes_to_the_true_temporal_maximum(self, tmp_path):
        """A real correctness check, not just a shape check: depth_max's own
        (dequantized) maximum across every node must equal the store's
        depth valid_max — both are, by definition, max over ALL (t, node)
        of depth. Proves the envelope pipeline computed a genuine max-over-
        time, not zeros or a copy of frame 0."""
        import zarr

        result = self._export(tmp_path)
        root = zarr.open_group(result["local_path"], mode="r")
        depth_max_attrs = root["depth_max"].attrs
        depth_attrs = root["depth"].attrs
        stored = root["depth_max"][:]
        dequantized = depth_max_attrs["offset"] + stored.astype(np.float32) * depth_max_attrs["scale"]
        assert abs(float(dequantized.max()) - depth_attrs["valid_max"]) < 1e-2


@requires_zarr
@pytest.mark.requires_anuga
class TestAdaptiveChunkLengthWrittenStore:
    """TASK-2719 AC2/AC3 (epic 2706 W8) — a WRITTEN store at prod scale, not
    just the pure derivation (TestDeriveChunkLengthT above).

    Real ANUGA runs at run-1328 scale (n_node=3,393,075) cannot be produced
    in a unit test, so ``_read_sww_arrays`` is monkeypatched to return a
    fabricated array set at exactly the D5 floor crossover (n_node=2,796,203)
    — small enough to allocate and gzip in-process, large enough to prove the
    exporter takes the adaptive chunk length rather than the old fixed 10.
    The exporter still needs ``anuga.config`` importable (for its g/rho_w/
    velocity_protection defaults) even though no simulation runs.
    """

    @staticmethod
    def _fabricate_sww_arrays(n_node, n_time=2):
        x = np.linspace(0.0, 100.0, n_node, dtype=np.float32)
        y = np.zeros(n_node, dtype=np.float32)
        # One triangle over the first three nodes — enough for
        # compute_inradius; the exporter never requires full mesh coverage.
        volumes = np.array([[0, 1, 2]], dtype=np.int32)
        elevation = np.zeros(n_node, dtype=np.float32)
        friction = np.full(n_node, 0.04, dtype=np.float32)
        stage = np.tile(elevation, (n_time, 1)).astype(np.float32)
        xmomentum = np.zeros((n_time, n_node), dtype=np.float32)
        ymomentum = np.zeros((n_time, n_node), dtype=np.float32)
        return dict(
            x=x, y=y, volumes=volumes, elevation=elevation, friction=friction,
            time=np.arange(n_time, dtype=np.float64), stage=stage,
            xmomentum=xmomentum, ymomentum=ymomentum,
            xllcorner=0.0, yllcorner=0.0, false_easting=0.0, false_northing=0.0,
            zone=55, anuga_version="0.0.0+test", revision_number="", revision_date="",
        )

    def _export_at_scale(self, tmp_path, n_node, n_time=2, run_id=1):
        fake_sww = self._fabricate_sww_arrays(n_node, n_time=n_time)
        with mock.patch.object(ps, "_read_sww_arrays", return_value=fake_sww):
            return ps.export_playback_store(
                input_data={
                    "run_label": f"run_synthetic_{run_id}",
                    "scenario_config": {
                        "project": 9, "id": 9, "run_id": run_id, "epsg": "EPSG:28355",
                    },
                },
                sww_path="unused-_read_sww_arrays-is-mocked",
                output_dir=str(tmp_path),
                upload=False,
            )

    def test_written_store_at_the_floor_crossover(self, tmp_path):
        """AC2 + AC3 combined (one export, both checks) — n_node is exactly
        the D5 floor crossover (2,796,203), so chunk_length_t must be 2 on
        ALL THREE quantized arrays AND declared in the group attrs alongside
        n_node/n_time."""
        import zarr

        n_node = 2_796_203
        n_time = 2
        result = self._export_at_scale(tmp_path, n_node, n_time=n_time, run_id=1)
        assert result["status"] == "ok", result

        root = zarr.open_group(result["local_path"], mode="r")
        # AC3 — the store declares its own dimensions.
        assert root.attrs["format_version"] == 2
        assert root.attrs["n_node"] == n_node
        assert root.attrs["n_time"] == n_time
        assert root.attrs["chunk_length_t"] == 2

        # AC2 — all three quantized arrays AGREE with the declared length.
        for name in ("depth", "x_velocity", "y_velocity"):
            arr = root[name]
            assert arr.chunks == (2, n_node), (
                f"{name}: chunks={arr.chunks}, expected (2, {n_node}) — "
                "quantized arrays must not drift from each other or from "
                "the declared chunk_length_t (playbackChunkShape.js refuses "
                "a store whose arrays disagree)"
            )

    def test_written_store_under_the_crossover_still_caps_at_ten(self, tmp_path):
        """A mesh just BELOW the CAP crossover (838,860) still gets the
        byte-identical-to-today chunk length of 10 — the adaptive rule does
        not touch small/typical-scale stores."""
        import zarr

        n_node = 500_000
        result = self._export_at_scale(tmp_path, n_node, run_id=2)
        root = zarr.open_group(result["local_path"], mode="r")
        assert root.attrs["chunk_length_t"] == 10
        assert root["depth"].chunks == (10, n_node)


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

    @pytest.mark.requires_anuga
    def test_v1_store_without_new_attrs_still_validates_clean(self, tmp_path, fixture_sww):
        """TASK-2719 AC4 backward compatibility — a v1 store (predating the
        n_node/n_time/chunk_length_t attrs and the adaptive chunk-length
        rule) must keep validating with ZERO violations FOREVER, including
        run 1328's real one — the store epic 2706's AC7 is measured against
        on prod. Simulated by exporting for real (this fixture's n_node is
        tiny, so the adaptive rule already gives chunk length 10 — the CAP,
        byte-identical to what a real pre-TASK-2719 export always wrote)
        then stripping the v2-only attrs and rolling format_version back to
        1 — exactly what a genuine pre-2719 store looks like on disk."""
        import zarr
        from run_anuga.validate_playback_store import validate_store

        input_data = {
            "run_label": "run_1_1_2",
            "scenario_config": {"project": 1, "id": 1, "run_id": 2, "epsg": "EPSG:28355"},
        }
        result = ps.export_playback_store(
            input_data=input_data, sww_path=str(fixture_sww),
            output_dir=str(tmp_path), upload=False,
        )
        root = zarr.open_group(result["local_path"], mode="a")
        assert root.attrs["chunk_length_t"] == 10, (
            "fixture mesh must stay small enough that CAP=10 applies, or "
            "this test stops simulating a real v1 store"
        )
        attrs = dict(root.attrs)
        attrs["format_version"] = 1
        del attrs["n_node"]
        del attrs["n_time"]
        del attrs["chunk_length_t"]
        root.attrs.put(attrs)

        assert validate_store(result["local_path"]) == []

    @pytest.mark.requires_anuga
    def test_v2_store_missing_new_attrs_is_caught(self, tmp_path, fixture_sww):
        """The gate is a real detector, not just permissive: a store that
        CLAIMS format_version=2 but is missing the new attrs must fail, not
        silently pass like a v1 store would."""
        import zarr
        from run_anuga.validate_playback_store import validate_store

        input_data = {
            "run_label": "run_1_1_3",
            "scenario_config": {"project": 1, "id": 1, "run_id": 3, "epsg": "EPSG:28355"},
        }
        result = ps.export_playback_store(
            input_data=input_data, sww_path=str(fixture_sww),
            output_dir=str(tmp_path), upload=False,
        )
        root = zarr.open_group(result["local_path"], mode="a")
        attrs = dict(root.attrs)
        assert attrs["format_version"] == 2
        del attrs["chunk_length_t"]
        root.attrs.put(attrs)

        violations = validate_store(result["local_path"])
        assert any("chunk_length_t" in v for v in violations), violations

    @pytest.mark.requires_anuga
    def test_v2_store_with_no_declared_envelopes_still_validates_clean(self, tmp_path, fixture_sww):
        """AC3/AC4 — a v2 store that declares NO envelopes (every store
        exported before TASK-2752, including run 1328's) must keep
        validating with ZERO violations. Simulated the same way the sibling
        v1 test above does: export for real, then strip the one new attr."""
        import zarr
        from run_anuga.validate_playback_store import validate_store

        input_data = {
            "run_label": "run_1_1_4",
            "scenario_config": {"project": 1, "id": 1, "run_id": 4, "epsg": "EPSG:28355"},
        }
        result = ps.export_playback_store(
            input_data=input_data, sww_path=str(fixture_sww),
            output_dir=str(tmp_path), upload=False,
        )
        root = zarr.open_group(result["local_path"], mode="a")
        attrs = dict(root.attrs)
        del attrs["envelope_quantities"]
        root.attrs.put(attrs)
        for name in ps.ENVELOPE_QUANTITIES:
            del root[f"{name}_max"]

        assert validate_store(result["local_path"]) == []

    @pytest.mark.requires_anuga
    def test_catches_a_declared_envelope_the_store_does_not_contain(self, tmp_path, fixture_sww):
        """AC3 — 'REJECTS a store that declares an envelope it does not
        contain'. A store that CLAIMS an envelope but has no backing array
        (or backing array for a name outside ENVELOPE_QUANTITIES) must fail,
        never silently pass."""
        import zarr
        from run_anuga.validate_playback_store import validate_store

        input_data = {
            "run_label": "run_1_1_5",
            "scenario_config": {"project": 1, "id": 1, "run_id": 5, "epsg": "EPSG:28355"},
        }
        result = ps.export_playback_store(
            input_data=input_data, sww_path=str(fixture_sww),
            output_dir=str(tmp_path), upload=False,
        )
        root = zarr.open_group(result["local_path"], mode="a")
        attrs = dict(root.attrs)
        attrs["envelope_quantities"] = list(attrs["envelope_quantities"]) + ["froude"]
        root.attrs.put(attrs)

        violations = validate_store(result["local_path"])
        assert any("froude" in v for v in violations), violations

    @pytest.mark.requires_anuga
    def test_catches_an_envelope_array_missing_a_quantization_attr(self, tmp_path, fixture_sww):
        """AC3 — a declared-and-present envelope array must still carry the
        full quantization attr set (schema §3); a real one with one stripped
        must be caught, exactly like the primitive quantity arrays are."""
        import zarr
        from run_anuga.validate_playback_store import validate_store

        input_data = {
            "run_label": "run_1_1_6",
            "scenario_config": {"project": 1, "id": 1, "run_id": 6, "epsg": "EPSG:28355"},
        }
        result = ps.export_playback_store(
            input_data=input_data, sww_path=str(fixture_sww),
            output_dir=str(tmp_path), upload=False,
        )
        root = zarr.open_group(result["local_path"], mode="a")
        arr = root["depth_max"]
        attrs = dict(arr.attrs)
        del attrs["scale"]
        arr.attrs.put(attrs)

        violations = validate_store(result["local_path"])
        assert any("depth_max" in v and "scale" in v for v in violations), violations


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
