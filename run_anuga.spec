# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for run-anuga CLI executable.

Build with: pyinstaller run_anuga.spec
Produces: dist/run-anuga/ (--onedir bundle)

The bundle includes rasterio, shapely, geopandas, and all simulation deps.
ANUGA is collected via collect_all() because meson-python's package structure
confuses PyInstaller's auto-discovery (it misses pure Python modules, only
collecting C extensions).
"""

import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

# rasterio ships PROJ and GDAL data files that must be bundled
try:
    import rasterio
    rasterio_datas = [
        (rasterio.gdal_data(), 'rasterio/gdal_data'),
        (rasterio.proj_data(), 'rasterio/proj_data'),
    ]
except Exception:
    rasterio_datas = []

# pyproj also ships PROJ data
try:
    import pyproj
    pyproj_datadir = pyproj.datadir.get_data_dir()
    pyproj_datas = [(pyproj_datadir, 'pyproj/proj_dir/share/proj')]
except Exception:
    pyproj_datas = []

# anuga: meson-python built package — must force-collect everything
anuga_datas, anuga_binaries, anuga_hiddenimports = collect_all('anuga')

a = Analysis(
    ['run_anuga/cli.py'],
    pathex=[],
    binaries=anuga_binaries,
    datas=rasterio_datas + pyproj_datas + [('examples', 'examples')] + anuga_datas,
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
    ] + anuga_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='run-anuga',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # CLI tool, not GUI
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='run-anuga',
)
