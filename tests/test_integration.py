"""Integration tests — require ANUGA and all simulation dependencies.

These are excluded from CI (--ignore=tests/test_integration.py).
Any test file that imports from run_anuga.run or run_anuga.run_utils
must go in this file, since those modules require heavy dependencies.
"""

import fnmatch
import shutil

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

    source_zip_input = Path(__file__).parent / "data" / zip_filename
    source_zip_target_dir = Path(__file__).parent / "data" / zip_filename.split('.')[0]
    print('start test_end_to_end_run')
    try:
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
    finally:
        print('finally')
        try:
            print(f'{source_zip_target_dir=}')
            shutil.rmtree(str(source_zip_target_dir))
        except FileNotFoundError:
            pass
