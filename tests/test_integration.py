import numpy
import os

from run_anuga.run_anuga.run import run_sim
from shutil import unpack_archive
from pathlib import Path

def test_end_to_end_run(tmp_path):
    # Unzip the test project (as downloaded from https://hydrata.com)
    package_directory = tmp_path / "package_directory"
    package_directory.mkdir()

    source_zip_path = str(Path(__file__).parent / "data" / "package_testbull_existing_2m_889.zip")
    unpack_archive(source_zip_path, package_directory)
    assert len(os.listdir(package_directory)) == 4

    source_editable_files_path = str(Path(__file__).parent / "data" / "package_testbull_existing_2m_889")
    result_zip_path = run_sim(source_editable_files_path)

    result_directory = tmp_path / "result_directory"
    result_directory.mkdir()
    unpack_archive(result_zip_path, result_directory)
    outputs_directory = result_directory / "outputs_344_256_889"
    assert len(os.listdir(outputs_directory)) == 6
