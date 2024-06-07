import fnmatch
import glob
import shutil

import pytest
import os

from run_anuga.run_anuga.run import run_sim
from run_anuga.run_anuga.run_utils import make_video
from shutil import unpack_archive
from pathlib import Path


@pytest.mark.parametrize(
    "zip_filename, package_dir_length, output_dir_name, result_directory_length", [
        # ("package_testbull_existing_2m_889.zip", 4, "outputs_344_256_889", 10),
        ("package_anuga_te_proposed_4_krzVHbr.zip", 4, "outputs_100_4_4", 10),
    ])
def test_end_to_end_run(tmp_path, zip_filename, package_dir_length, output_dir_name, result_directory_length):
    source_zip_input = Path(__file__).parent / "data" / zip_filename
    source_zip_target = Path(__file__).parent / "data" / zip_filename.split('.')[0]
    unpack_archive(str(source_zip_input), str(source_zip_target))
    assert len(os.listdir(str(source_zip_target))) == package_dir_length

    result_directory = source_zip_target / output_dir_name
    run_sim(str(source_zip_target))
    result_directory.mkdir(exist_ok=True)
    result_filenames = os.listdir(result_directory)
    for file_name in [
        'run_anuga.log',
        'run_*_velocity_max.tif',
        'run_*.msh',
        'run_*.sww',
        'run_*_depth_max.tif',
        'run_*_depthIntegratedVelocity_max.tif'
    ]:
        assert fnmatch.filter(result_filenames, file_name)

    try:
        shutil.rmtree(source_zip_target)
    except FileNotFoundError:
        pass

@pytest.mark.skip
@pytest.mark.parametrize("still_images_directory, run_label, video_type", [
    ('../run_anuga/tests/data/video/depthIntegratedVelocity', "run_402_324_1062", "depthIntegratedVelocity")
])
def test_make_video(still_images_directory, run_label, video_type):
    full_directory = os.path.join(os.getcwd(), still_images_directory)
    make_video(str(full_directory), run_label, video_type)

    assert len(glob.glob(f"{still_images_directory}/{run_label}_{video_type}.mp4")) == 1
