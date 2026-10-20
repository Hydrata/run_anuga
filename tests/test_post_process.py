"""Integration tests for post_process_sww().

Requires ANUGA and all simulation dependencies installed.
Tests GeoTIFF generation from SWW output files.
"""

import shutil
from unittest import mock

import pytest

try:
    import zarr  # noqa: F401

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

requires_zarr = pytest.mark.skipif(not HAS_ZARR, reason="zarr not installed")


@pytest.mark.requires_anuga
@pytest.mark.slow
class TestPostProcess:
    @pytest.fixture(autouse=True)
    def _run_simulation(self, small_test_copy):
        """Run simulation once, then test post-processing."""
        from run_anuga.run import run_sim

        run_sim(str(small_test_copy))
        self.package_dir = small_test_copy

    def test_post_process_creates_tiffs(self):
        from run_anuga.run_utils import post_process_sww

        post_process_sww(str(self.package_dir))
        tifs = list(self.package_dir.glob("outputs_*/*_max.tif"))
        assert len(tifs) >= 2

    def test_post_process_tiff_has_valid_crs(self):
        rasterio = pytest.importorskip("rasterio")
        from run_anuga.run_utils import post_process_sww

        post_process_sww(str(self.package_dir))
        tifs = list(self.package_dir.glob("outputs_*/*depth_max.tif"))
        assert len(tifs) >= 1
        with rasterio.open(str(tifs[0])) as ds:
            assert ds.crs is not None

    def test_post_process_all_quantities_present(self):
        from run_anuga.run_utils import post_process_sww

        post_process_sww(str(self.package_dir))
        tifs = list(self.package_dir.glob("outputs_*/*_max.tif"))
        stems = {f.stem for f in tifs}
        # All 4 output quantities must produce max TIFFs
        assert any("depth" in s for s in stems)
        assert any("velocity" in s for s in stems)
        assert any("depthIntegratedVelocity" in s for s in stems)
        assert any("stage" in s for s in stems)

    def test_post_process_idempotent(self):
        """Running post-process twice doesn't error."""
        from run_anuga.run_utils import post_process_sww

        post_process_sww(str(self.package_dir))
        # Second call should overwrite cleanly
        post_process_sww(str(self.package_dir))
        tifs = list(self.package_dir.glob("outputs_*/*_max.tif"))
        assert len(tifs) >= 2


@pytest.mark.requires_anuga
class TestPostProcessLegacyTifsFlagGate:
    """TASK-2622 (W1.1, epic 2618) — the per-timestep myTimeStep='all'
    Make_Geotif pass is REMOVED by default (replaced by the playback store);
    a run only regenerates it via the explicit legacy_per_timestep_tifs=True
    opt-in (CLI: --legacy-per-timestep-tifs). The myTimeStep='max' pass is
    UNCHANGED either way. Uses a real (fixture) SWW with Make_Geotif MOCKED
    so this stays fast — no live ANUGA simulation.
    """

    @pytest.fixture
    def package_with_sww(self, scenario_package, fixture_sww):
        """scenario_package (project=1, id=1, run_id=1) + a copy of the real
        session-scoped fixture SWW placed where post_process_sww expects it."""
        output_dir = scenario_package / "outputs_1_1_1"
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(fixture_sww, output_dir / "run_1_1_1.sww")
        return scenario_package

    def _call_post_process(self, package_dir, **kwargs):
        from run_anuga.run_utils import post_process_sww

        with mock.patch("anuga.utilities.plot_utils.Make_Geotif") as mock_geotif, \
             mock.patch("run_anuga.playback_store.export_playback_store") as mock_playback:
            mock_playback.return_value = {"status": "ok"}
            post_process_sww(str(package_dir), **kwargs)
        return mock_geotif, mock_playback

    def test_default_no_all_pass(self, package_with_sww):
        mock_geotif, _ = self._call_post_process(package_with_sww)
        time_steps = [c.kwargs.get("myTimeStep") for c in mock_geotif.call_args_list]
        assert "all" not in time_steps, "myTimeStep='all' must NOT run by default (TASK-2622)"
        assert "max" in time_steps, "myTimeStep='max' must be unchanged"

    def test_legacy_flag_restores_all_pass(self, package_with_sww):
        mock_geotif, _ = self._call_post_process(
            package_with_sww, legacy_per_timestep_tifs=True
        )
        time_steps = [c.kwargs.get("myTimeStep") for c in mock_geotif.call_args_list]
        assert "all" in time_steps
        assert "max" in time_steps

    def test_max_pass_unchanged_regardless_of_flag(self, package_with_sww):
        """Exactly one 'max' call either way — the flag only toggles 'all'."""
        for flag in (False, True):
            mock_geotif, _ = self._call_post_process(
                package_with_sww, legacy_per_timestep_tifs=flag
            )
            time_steps = [c.kwargs.get("myTimeStep") for c in mock_geotif.call_args_list]
            assert time_steps.count("max") == 1

    def test_playback_store_export_invoked(self, package_with_sww):
        _, mock_playback = self._call_post_process(package_with_sww)
        mock_playback.assert_called_once()

    def test_playback_store_export_receives_sww_path_and_output_dir(self, package_with_sww):
        _, mock_playback = self._call_post_process(package_with_sww)
        _, call_kwargs = mock_playback.call_args
        assert call_kwargs["sww_path"].endswith("run_1_1_1.sww")
        assert call_kwargs["output_dir"] == str(package_with_sww / "outputs_1_1_1")

    def test_playback_features_default_on(self, package_with_sww, monkeypatch):
        monkeypatch.delenv("RUN_ANUGA_PLAYBACK_SPATIAL_ORDER", raising=False)
        monkeypatch.delenv("RUN_ANUGA_PLAYBACK_TEMPORAL_DELTA", raising=False)
        _, mock_playback = self._call_post_process(package_with_sww)
        _, call_kwargs = mock_playback.call_args
        assert call_kwargs["spatial_order"] is True
        assert call_kwargs["temporal_delta"] is True

    @pytest.mark.parametrize("order_env,delta_env", [("0", "0"), ("false", "1"), ("1", "off")])
    def test_playback_kill_switches_read_from_env(self, package_with_sww, monkeypatch, order_env, delta_env):
        """Epic 2981 merge: a Batch jobdef env var switches either feature off
        at the run's own call site, without a rebake."""
        monkeypatch.setenv("RUN_ANUGA_PLAYBACK_SPATIAL_ORDER", order_env)
        monkeypatch.setenv("RUN_ANUGA_PLAYBACK_TEMPORAL_DELTA", delta_env)
        _, mock_playback = self._call_post_process(package_with_sww)
        _, call_kwargs = mock_playback.call_args
        assert call_kwargs["spatial_order"] is (order_env == "1")
        assert call_kwargs["temporal_delta"] is (delta_env == "1")

    def test_domain_threaded_through_to_playback_export(self, package_with_sww):
        from types import SimpleNamespace

        fake_domain = SimpleNamespace(minimum_allowed_height=1e-12, flow_algorithm="DE1")
        _, mock_playback = self._call_post_process(package_with_sww, domain=fake_domain)
        _, call_kwargs = mock_playback.call_args
        assert call_kwargs["domain"] is fake_domain


@pytest.mark.requires_anuga
class TestPostProcessRealPlaybackExport:
    """Same fixture, but lets the REAL playback exporter run (Make_Geotif
    still mocked out — this class only proves the wiring produces a real,
    schema-conformant store end to end from post_process_sww's own call
    site, not just that export_playback_store is called)."""

    @pytest.fixture
    def package_with_sww(self, scenario_package, fixture_sww):
        output_dir = scenario_package / "outputs_1_1_1"
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(fixture_sww, output_dir / "run_1_1_1.sww")
        return scenario_package

    @requires_zarr
    def test_real_playback_store_produced_and_valid(self, package_with_sww):
        from run_anuga.run_utils import post_process_sww
        from run_anuga.validate_playback_store import validate_store

        with mock.patch("anuga.utilities.plot_utils.Make_Geotif"):
            post_process_sww(str(package_with_sww))

        stores = list((package_with_sww / "outputs_1_1_1").glob("*_playback.zarr"))
        assert len(stores) == 1
        assert validate_store(stores[0]) == []


@pytest.mark.requires_anuga
class TestPostProcessTruePeaks:
    """TASK-3772 (epic 3585) — when the SWW carries the engine's per-step
    max arrays, post_process_sww grids them through Make_Geotif's xyz path
    and folds them into the same-named *_max.tif rasters. Make_Geotif is
    mocked (or faked with tiny real rasters) so this stays fast."""

    QUANTITIES = ("depth", "velocity", "depthIntegratedVelocity", "stage")

    def _package(self, scenario_package, sww):
        output_dir = scenario_package / "outputs_1_1_1"
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(sww, output_dir / "run_1_1_1.sww")
        return scenario_package

    def _run(self, package_dir, side_effect=None):
        from run_anuga.run_utils import post_process_sww

        with mock.patch("anuga.utilities.plot_utils.Make_Geotif", side_effect=side_effect) as mock_geotif, \
             mock.patch("run_anuga.playback_store.export_playback_store") as mock_playback:
            mock_playback.return_value = {"status": "ok"}
            post_process_sww(str(package_dir))
        return mock_geotif

    def test_xyz_path_used_once_per_quantity_when_engine_arrays_exist(self, scenario_package, fixture_sww):
        import numpy as np

        from run_anuga import peaks

        pkg = self._package(scenario_package, fixture_sww)
        mock_geotif = self._run(pkg)
        calls = mock_geotif.call_args_list
        max_calls = [c for c in calls if c.kwargs.get("myTimeStep") == "max"]
        xyz_calls = [c for c in calls if isinstance(c.kwargs.get("swwFile"), np.ndarray)]
        assert len(max_calls) == 1
        assert [c.kwargs["output_quantities"] for c in xyz_calls] == [[q] for q in self.QUANTITIES]
        engine = peaks.read_engine_peaks(fixture_sww)
        n_tri = engine["max_depth_c"].shape[0]
        base = max_calls[0].kwargs
        for c, q in zip(xyz_calls, self.QUANTITIES):
            xyz = c.kwargs["swwFile"]
            assert xyz.shape == (n_tri, 3)
            np.testing.assert_allclose(xyz[:, 2], engine[peaks.RASTER_QUANTITY_TO_ENGINE[q]])
            for key in ("CellSize", "EPSG_CODE", "bounding_polygon", "internal_holes",
                        "k_nearest_neighbours", "min_allowed_height"):
                assert c.kwargs[key] == base[key], key

    def test_old_sww_keeps_only_the_frame_max_pass(self, scenario_package, fixture_sww_frames_only):
        pkg = self._package(scenario_package, fixture_sww_frames_only)
        mock_geotif = self._run(pkg)
        assert [c.kwargs.get("myTimeStep") for c in mock_geotif.call_args_list] == ["max"]

    @staticmethod
    def _write_tif(path, value):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        data = np.full((3, 4), value, dtype=np.float32)
        data[0, 0] = np.nan  # outside the bounding polygon
        with rasterio.open(path, "w", driver="GTiff", height=3, width=4, count=1, dtype="float32",
                           crs="EPSG:28355", transform=from_origin(0, 3, 1, 1), nodata=np.nan) as ds:
            ds.write(data, 1)

    def _fake_geotif(self, engine_values):
        def fake(**kwargs):
            out = kwargs["output_dir"]
            if kwargs.get("myTimeStep") == "max":
                for q in kwargs["output_quantities"]:
                    self._write_tif(f"{out}/run_1_1_1_{q}_max.tif", 1.0)
            else:
                q = kwargs["output_quantities"][0]
                self._write_tif(f"{out}/PointData_{q}.tif", engine_values[q])
        return fake

    def test_rasters_combine_never_below_frames_and_div_is_replaced(self, scenario_package, fixture_sww):
        import math

        import rasterio

        engine_values = {"depth": 2.0, "velocity": 0.5, "stage": 0.5, "depthIntegratedVelocity": 0.25}
        pkg = self._package(scenario_package, fixture_sww)
        self._run(pkg, side_effect=self._fake_geotif(engine_values))
        out = pkg / "outputs_1_1_1"
        expected = {"depth": 2.0, "velocity": 1.0, "stage": 1.0, "depthIntegratedVelocity": 0.25}
        for q, value in expected.items():
            with rasterio.open(out / f"run_1_1_1_{q}_max.tif") as ds:
                band = ds.read(1)
                assert math.isnan(ds.nodata)
                assert ds.tags()["peak_basis"] == "engine_per_step"
            assert math.isnan(band[0, 0])
            assert float(band[1, 1]) == value, q
        assert not list(out.glob("true_peaks_*")), "temp dir left behind"
        assert not list(out.glob("PointData_*")), "xyz rasters must not leak into outputs"

    def test_frame_rasters_are_labelled_saved_frames(self, scenario_package, fixture_sww_frames_only):
        import rasterio

        pkg = self._package(scenario_package, fixture_sww_frames_only)
        self._run(pkg, side_effect=self._fake_geotif({}))
        for q in self.QUANTITIES:
            with rasterio.open(pkg / "outputs_1_1_1" / f"run_1_1_1_{q}_max.tif") as ds:
                assert ds.tags()["peak_basis"] == "saved_frames"
                assert ds.tags()["peak_basis_label"] == "max over saved frames"
                assert float(ds.read(1)[1, 1]) == 1.0
