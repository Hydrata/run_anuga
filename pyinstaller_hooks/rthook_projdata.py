"""PyInstaller runtime hook: set PROJ_DATA and GDAL_DATA for rasterio/pyproj."""
import os
import sys

# In a PyInstaller bundle, sys._MEIPASS is the path to _internal/
bundle_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

proj_dir = os.path.join(bundle_dir, 'rasterio', 'proj_data')
if os.path.isdir(proj_dir):
    os.environ.setdefault('PROJ_DATA', proj_dir)
    os.environ.setdefault('PROJ_LIB', proj_dir)

gdal_dir = os.path.join(bundle_dir, 'rasterio', 'gdal_data')
if os.path.isdir(gdal_dir):
    os.environ.setdefault('GDAL_DATA', gdal_dir)

# pyproj may also have its own proj data
pyproj_dir = os.path.join(bundle_dir, 'pyproj', 'proj_dir', 'share', 'proj')
if os.path.isdir(pyproj_dir):
    os.environ.setdefault('PROJ_DATA', pyproj_dir)
    os.environ.setdefault('PROJ_LIB', pyproj_dir)
