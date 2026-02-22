"""PyInstaller runtime hook: set PROJ_DATA and GDAL_DATA for rasterio/pyproj.

Rasterio bundles its own PROJ data that matches its PROJ library version.
Pyproj may bundle an older version. We prefer rasterio's data to avoid
"DATABASE.LAYOUT.VERSION.MINOR" mismatch errors.
"""
import os
import sys

# In a PyInstaller bundle, sys._MEIPASS is the path to _internal/
bundle_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

# Prefer rasterio's PROJ data (matches its bundled PROJ library)
rasterio_proj = os.path.join(bundle_dir, 'rasterio', 'proj_data')
pyproj_proj = os.path.join(bundle_dir, 'pyproj', 'proj_dir', 'share', 'proj')

# Use rasterio's data if available, else fall back to pyproj's
if os.path.isdir(rasterio_proj):
    os.environ['PROJ_DATA'] = rasterio_proj
    os.environ['PROJ_LIB'] = rasterio_proj
elif os.path.isdir(pyproj_proj):
    os.environ['PROJ_DATA'] = pyproj_proj
    os.environ['PROJ_LIB'] = pyproj_proj

gdal_dir = os.path.join(bundle_dir, 'rasterio', 'gdal_data')
if os.path.isdir(gdal_dir):
    os.environ['GDAL_DATA'] = gdal_dir
