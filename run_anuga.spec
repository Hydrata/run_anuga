# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for run-anuga CLI executable.

Build with: pyinstaller run_anuga.spec
Produces: dist/run-anuga/ (--onedir bundle)

The bundle includes rasterio, shapely, geopandas, and all simulation deps.
ANUGA's C extensions are included via hidden imports.
"""

import os
import sys

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

a = Analysis(
    ['run_anuga/cli.py'],
    pathex=[],
    binaries=[],
    datas=rasterio_datas + pyproj_datas,
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
        # anuga
        'anuga',
        'anuga.utilities.plot_utils',
        'anuga.utilities.spatialInputUtil',
        'anuga.file_conversion.tif2array',
        'anuga.file_conversion.tif2point_values',
        # scipy (used by anuga)
        'scipy.spatial',
        'scipy.interpolate',
        'scipy.sparse',
        # numpy
        'numpy',
        # netCDF4
        'netCDF4',
        'cftime',
    ],
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
