"""Integration tests for checkpoint/restart functionality.

Requires ANUGA installed with dill for pickling.
"""

import pytest


@pytest.mark.requires_anuga
@pytest.mark.slow
class TestCheckpoint:
    def test_checkpoint_files_created(self, small_test_copy):
        """Batch 1 creates checkpoint pickle files."""
        from run_anuga.run import run_sim

        run_sim(str(small_test_copy), batch_number=1)
        pickles = list(small_test_copy.glob("**/checkpoints/*.pickle"))
        assert len(pickles) >= 1

    def test_checkpoint_directory_exists(self, small_test_copy):
        """Checkpoint directory is created during simulation."""
        from run_anuga.run import run_sim

        run_sim(str(small_test_copy), batch_number=1)
        checkpoint_dirs = list(small_test_copy.glob("outputs_*/checkpoints"))
        assert len(checkpoint_dirs) == 1
        assert checkpoint_dirs[0].is_dir()
