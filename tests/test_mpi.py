"""MPI parallel execution tests.

These tests MUST be run via:
    mpirun -np 2 python -m pytest tests/test_mpi.py -v

They test that ANUGA's MPI-based domain distribution and
SWW merge work correctly with multiple processes.
"""

import pytest


@pytest.mark.requires_anuga
@pytest.mark.mpi
@pytest.mark.slow
class TestMPIParallel:
    def test_parallel_run_produces_output(self, small_test_copy):
        """2-process MPI run produces merged SWW and GeoTIFFs."""
        from run_anuga.run import run_sim

        run_sim(str(small_test_copy))

        sww_files = list(small_test_copy.glob("outputs_*/*.sww"))
        assert len(sww_files) >= 1

    def test_parallel_produces_geotiffs(self, small_test_copy):
        """Parallel run produces depth and velocity max GeoTIFFs."""
        from run_anuga.run import run_sim

        run_sim(str(small_test_copy))

        tif_files = list(small_test_copy.glob("outputs_*/*_max.tif"))
        assert len(tif_files) >= 2
        names = {f.stem for f in tif_files}
        assert any("depth" in n for n in names)
        assert any("velocity" in n for n in names)

    def test_parallel_log_file(self, small_test_copy):
        """MPI run creates log file on rank 0."""
        from run_anuga.run import run_sim

        run_sim(str(small_test_copy))

        log_files = list(small_test_copy.glob("outputs_*/run_anuga_*.log"))
        assert len(log_files) >= 1

    def test_parallel_checkpoint_per_rank(self, small_test_copy):
        """MPI run creates checkpoint files for each rank."""
        from run_anuga.run import run_sim
        import anuga

        run_sim(str(small_test_copy), batch_number=1)

        if anuga.myid == 0:
            pickles = list(small_test_copy.glob("**/checkpoints/*.pickle"))
            # Should have at least one per process
            assert len(pickles) >= anuga.numprocs
