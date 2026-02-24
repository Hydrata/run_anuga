"""Integration tests for run_sim() lifecycle.

Requires ANUGA and all simulation dependencies installed.
These tests use the examples/small_test scenario.
"""

import pytest


@pytest.mark.requires_anuga
@pytest.mark.slow
class TestSimLifecycle:
    def test_run_sim_produces_sww(self, small_test_copy):
        """Full simulation produces .sww file."""
        from run_anuga.run import run_sim

        run_sim(str(small_test_copy))
        sww_files = list(small_test_copy.glob("outputs_*/*.sww"))
        assert len(sww_files) >= 1

    def test_run_sim_produces_geotiffs(self, small_test_copy):
        """Full simulation produces depth + velocity GeoTIFFs."""
        from run_anuga.run import run_sim

        run_sim(str(small_test_copy))
        tif_files = list(small_test_copy.glob("outputs_*/*_max.tif"))
        assert len(tif_files) >= 2
        names = {f.stem for f in tif_files}
        assert any("depth" in n for n in names)
        assert any("velocity" in n for n in names)

    def test_run_sim_log_file(self, small_test_copy):
        """Simulation creates log file with completion message."""
        from run_anuga.run import run_sim

        run_sim(str(small_test_copy))
        log_files = list(small_test_copy.glob("outputs_*/run_anuga_*.log"))
        assert len(log_files) == 1
        content = log_files[0].read_text()
        assert "finished run:" in content  # run.py logs this at completion

    def test_run_sim_with_callback(self, small_test_copy):
        """Callback receives status updates during simulation."""
        from run_anuga.run import run_sim

        statuses = []

        class RecordingCallback:
            def on_status(self, status, **kw):
                statuses.append(status)

            def on_metric(self, key, value):
                pass

            def on_file(self, key, filepath):
                pass

        run_sim(str(small_test_copy), callback=RecordingCallback())
        assert len(statuses) > 0
        # Should have percentage updates
        assert any("%" in s for s in statuses)

    def test_run_sim_output_directory_structure(self, small_test_copy):
        """Output directory has expected structure."""
        from run_anuga.run import run_sim

        run_sim(str(small_test_copy))
        output_dirs = list(small_test_copy.glob("outputs_*"))
        assert len(output_dirs) == 1
        output_dir = output_dirs[0]
        # Should contain: sww, log, tifs, checkpoint dir
        assert any(f.suffix == ".sww" for f in output_dir.iterdir())
        assert any(f.suffix == ".log" for f in output_dir.iterdir())
        assert (output_dir / "checkpoints").is_dir()
