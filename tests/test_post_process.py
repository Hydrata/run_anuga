"""Integration tests for post_process_sww().

Requires ANUGA and all simulation dependencies installed.
Tests GeoTIFF generation from SWW output files.
"""

import os
import shutil
from unittest import mock

import pytest

FIXTURE_SWW = os.path.join(os.path.dirname(__file__), "..", "domain.sww")


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
    def package_with_sww(self, scenario_package):
        """scenario_package (project=1, id=1, run_id=1) + a copy of the real
        domain.sww fixture placed where post_process_sww expects it."""
        output_dir = scenario_package / "outputs_1_1_1"
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIXTURE_SWW, output_dir / "run_1_1_1.sww")
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
    def package_with_sww(self, scenario_package):
        output_dir = scenario_package / "outputs_1_1_1"
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIXTURE_SWW, output_dir / "run_1_1_1.sww")
        return scenario_package

    def test_real_playback_store_produced_and_valid(self, package_with_sww):
        zarr = pytest.importorskip("zarr")  # noqa: F841
        from run_anuga.run_utils import post_process_sww
        from run_anuga.validate_playback_store import validate_store

        with mock.patch("anuga.utilities.plot_utils.Make_Geotif"):
            post_process_sww(str(package_with_sww))

        stores = list((package_with_sww / "outputs_1_1_1").glob("*_playback.zarr"))
        assert len(stores) == 1
        assert validate_store(stores[0]) == []
