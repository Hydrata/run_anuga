"""Integration tests for post_process_sww().

Requires ANUGA and all simulation dependencies installed.
Tests GeoTIFF generation from SWW output files.
"""

import pytest


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
        # Check at least depth and velocity max tiffs
        assert any("depth" in s for s in stems)
        assert any("velocity" in s for s in stems)

    def test_post_process_idempotent(self):
        """Running post-process twice doesn't error."""
        from run_anuga.run_utils import post_process_sww

        post_process_sww(str(self.package_dir))
        # Second call should overwrite cleanly
        post_process_sww(str(self.package_dir))
        tifs = list(self.package_dir.glob("outputs_*/*_max.tif"))
        assert len(tifs) >= 2
