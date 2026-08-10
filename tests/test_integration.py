"""Integration tests — require ANUGA and all simulation dependencies.

These RUN in CI: they carry ``@pytest.mark.requires_anuga`` and are executed by
the REQUIRED ``test-e2e`` physics gate (.github/workflows/ci.yml, `pytest tests/
-v -m requires_anuga --timeout=1800`), which force-installs the Hydrata
anuga_core fork first. They are deselected only from the light `test` job and
the `test-geo` job, whose marker filters exclude ``requires_anuga``; locally
tests/conftest.py auto-skips them when ``import anuga`` fails. (This docstring
used to claim "excluded from CI (--ignore=tests/test_integration.py)" — that
predates the marker-based gating and was false.)

Any test file that imports from run_anuga.run or run_anuga.run_utils
must go in this file, since those modules require heavy dependencies.
"""

import fnmatch

import pytest
import os

from shutil import unpack_archive
from pathlib import Path


@pytest.mark.requires_anuga
@pytest.mark.parametrize(
    "zip_filename, package_dir_length, output_dir_name, result_directory_length", [
        # Merewether Urban Flood Benchmark — canonical goto test run (2007 Pasha Bulka event).
        # Lean package: inputs/ + scenario.json only; run_sim regenerates the mesh into outputs_1_1_1/.
        # We deliberately do NOT store large pre-baked run-packages (mesh/sww/anuga_repo) in git;
        # the package_* zips were removed in favour of this self-regenerating fixture.
        ("merewether_package.zip", 2, "outputs_1_1_1", 10),
    ])
def test_end_to_end_run(tmp_path, zip_filename, package_dir_length, output_dir_name, result_directory_length):
    # Import at test time, not module level, so the test file can be collected
    # by pytest even when ANUGA is not installed.
    from run_anuga.run import run_sim

    # The .zip is a tracked, read-only fixture, so it is read from the repo; the
    # sim, however, unpacks and runs under `tmp_path`, NOT under tests/data/.
    # Previously the destination was `tests/data/merewether_package/` — inside
    # the working tree — and the only cleanup was a `finally: shutil.rmtree`.
    # A `finally` is lost to a hard kill (SIGKILL/OOM, cancelled job, crashed
    # box) mid-sim, and `unpack_archive` does not clear a non-empty destination,
    # so a surviving `outputs_1_1_1/` made the NEXT local run fail `assert 3 ==
    # 2` below — then that run's own `finally` deleted the leftover, making it a
    # one-shot, self-healing, non-reproducible red. `tmp_path` is a fresh
    # per-test directory, so the destination is empty by construction and the
    # working tree is never written to. Same convention as
    # tests/test_e2e_merewether.py and conftest's `fixture_sww`, which already
    # run the real sim out of tmp_path in this very CI job.
    #
    # Dropping the `finally` is deliberate: pytest retains the last 3 tmp_path
    # trees, so a FAILED e2e run now leaves its .sww/.tif/.msh/.log on disk for
    # post-mortem instead of rmtree-ing the evidence.
    source_zip_input = Path(__file__).parent / "data" / zip_filename
    source_zip_target_dir = tmp_path / zip_filename.split('.')[0]
    print('start test_end_to_end_run')
    unpack_archive(str(source_zip_input), str(source_zip_target_dir))
    assert len(os.listdir(str(source_zip_target_dir))) == package_dir_length

    result_directory = source_zip_target_dir / output_dir_name
    run_sim(str(source_zip_target_dir))
    result_directory.mkdir(exist_ok=True)
    result_filenames = os.listdir(result_directory)
    for file_name in [
        'run_anuga_*.log',
        'run_*_velocity_max.tif',
        'run_*.msh',
        'run_*.sww',
        'run_*_depth_max.tif',
        'run_*_depthIntegratedVelocity_max.tif'
    ]:
        assert fnmatch.filter(result_filenames, file_name)
