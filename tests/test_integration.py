import netCDF4
import os

from run_anuga.run_anuga.run import run_sim
from shutil import unpack_archive

def test_end_to_end_run(tmp_path):
    # Unzip the test project (as downloaded from https://hydrata.com)
    package_directory = tmp_path / "package_directory"
    package_directory.mkdir()
    unpack_archive("./data/package_testbull_existing_2m_889.zip", package_directory)
    assert len(os.listdir(package_directory)) == 4

    import sys
    print(sys.version)
    print(sys.executable)
    import netCDF4
    try:
        from pip._internal.operations import freeze
    except ImportError:  # pip < 10.0
        from pip.operations import freeze
    pkgs = freeze.freeze()
    for pkg in pkgs: print(pkg)


    run_sim(package_directory)

    assert True