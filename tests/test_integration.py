import shutil

import pytest
import os

from run_anuga.run import run_sim
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
    result_zip_path = run_sim(str(source_zip_target))
    result_directory.mkdir(exist_ok=True)
    unpack_archive(result_zip_path, result_directory)
    try:
        os.remove(result_zip_path)
    except FileNotFoundError:
        pass
    assert len(os.listdir(result_directory)) == result_directory_length

    try:
        shutil.rmtree(source_zip_target)
    except FileNotFoundError:
        pass
