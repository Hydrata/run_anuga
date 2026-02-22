# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for run-anuga CLI executable.

Build with: pyinstaller run_anuga.spec
Produces: dist/run-anuga (single-file executable)

The bundle includes rasterio, shapely, geopandas, and all simulation deps.
ANUGA is collected via importlib path discovery because meson-python's package
structure confuses PyInstaller's collect_all() on Windows.
"""

import importlib
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

# --- rasterio PROJ and GDAL data files ---
rasterio_datas = []
try:
    import rasterio
    pkg_dir = os.path.dirname(rasterio.__file__)
    for subdir, dest in [('gdal_data', 'rasterio/gdal_data'),
                          ('proj_data', 'rasterio/proj_data')]:
        data_path = os.path.join(pkg_dir, subdir)
        if os.path.isdir(data_path):
            rasterio_datas.append((data_path, dest))
except Exception:
    pass

# --- pyproj PROJ data (fallback if rasterio's is missing) ---
pyproj_datas = []
try:
    import pyproj
    pyproj_datadir = pyproj.datadir.get_data_dir()
    if os.path.isdir(pyproj_datadir):
        pyproj_datas = [(pyproj_datadir, 'pyproj/proj_dir/share/proj')]
except Exception:
    pass

# --- anuga: meson-python built package ---
# collect_all('anuga') works on Linux but fails on Windows with
# "skipping data collection for module 'anuga' as it is not a package."
# Workaround: locate the package via importlib and use os.walk to include
# the entire directory, plus collect_submodules for hidden imports.
anuga_datas = []
anuga_binaries = []
anuga_hiddenimports = []

try:
    anuga_datas, anuga_binaries, anuga_hiddenimports = collect_all('anuga')
    print(f"collect_all('anuga'): {len(anuga_datas)} datas, "
          f"{len(anuga_binaries)} binaries, {len(anuga_hiddenimports)} hiddenimports")
except Exception as e:
    print(f"collect_all('anuga') failed: {e}")

# If collect_all returned nothing useful, fall back to manual discovery
if not anuga_datas and not anuga_binaries:
    print("collect_all('anuga') returned empty — using importlib fallback")
    try:
        spec = importlib.util.find_spec('anuga')
        if spec and spec.origin:
            anuga_pkg_dir = os.path.dirname(spec.origin)
            print(f"Found anuga at: {anuga_pkg_dir}")
            # Walk the entire anuga directory tree and add all files
            for dirpath, dirnames, filenames in os.walk(anuga_pkg_dir):
                for fn in filenames:
                    full = os.path.join(dirpath, fn)
                    rel = os.path.relpath(full, os.path.dirname(anuga_pkg_dir))
                    dest_dir = os.path.dirname(rel)
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in ('.pyd', '.so', '.dll', '.dylib'):
                        anuga_binaries.append((full, dest_dir))
                    else:
                        anuga_datas.append((full, dest_dir))
            print(f"Manual collection: {len(anuga_datas)} datas, "
                  f"{len(anuga_binaries)} binaries")
        # Hidden imports — collect all submodules
        anuga_hiddenimports = collect_submodules('anuga')
        print(f"collect_submodules('anuga'): {len(anuga_hiddenimports)} imports")
    except Exception as e:
        print(f"WARNING: Could not locate anuga package: {e}")

a = Analysis(
    ['run_anuga/cli.py'],
    pathex=[],
    binaries=anuga_binaries,
    datas=rasterio_datas + pyproj_datas + anuga_datas,
    hiddenimports=[
        # rasterio internals
        'rasterio._shim',
        'rasterio.control',
        'rasterio.crs',
        'rasterio.sample',
        'rasterio.vrt',
        'rasterio._features',
        'rasterio.features',
        'rasterio.warp',
        'rasterio.mask',
        'rasterio.transform',
        'rasterio.enums',
        # fiona (used by geopandas)
        'fiona',
        'fiona.schema',
        # geopandas + pandas
        'geopandas',
        'pandas',
        'pyogrio',
        'pyproj',
        # shapely
        'shapely',
        'shapely.geometry',
        # scipy (used by anuga)
        'scipy.spatial',
        'scipy.interpolate',
        'scipy.sparse',
        # numpy
        'numpy',
        # netCDF4
        'netCDF4',
        'cftime',
        # undeclared anuga runtime deps
        'six',
        'dill',
        'mpi4py',
        'triangle',
        'matplotlib',
    ] + anuga_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyinstaller_hooks/rthook_projdata.py'],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='run-anuga',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
