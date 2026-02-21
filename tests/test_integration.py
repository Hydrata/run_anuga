import fnmatch
import glob
import shutil

import pytest
import os

from run_anuga.run import run_sim
from run_anuga.run_utils import make_video
from shutil import unpack_archive
from pathlib import Path


@pytest.mark.parametrize(
    "zip_filename, package_dir_length, output_dir_name, result_directory_length", [
        ("package_fourthreextwo_halves_checkpoint_173.zip", 4, "outputs_18_17_173", 10),
        # ("package_anuga_te_proposed_4_krzVHbr.zip", 4, "outputs_100_4_4", 10),
    ])
def test_end_to_end_run(tmp_path, zip_filename, package_dir_length, output_dir_name, result_directory_length):
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
