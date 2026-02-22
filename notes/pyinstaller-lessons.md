# PyInstaller Lessons Learned

## Onefile vs Onedir

- `--onedir` (default): produces `dist/run-anuga/` with exe + `_internal/` directory. Fragile — users must keep them together.
- `--onefile`: produces single `dist/run-anuga` executable. Extracts to temp dir at runtime. ~1-2s cold start but much simpler.
- In the spec, onefile means: include `a.binaries` and `a.datas` in `EXE()`, remove `COLLECT()` block entirely.

## meson-python packages (anuga)

PyInstaller's `collect_all()` doesn't work for meson-python built packages on Windows:
```
WARNING: collect_data_files - skipping data collection for module 'anuga' as it is not a package.
WARNING: collect_dynamic_libs - skipping library collection for module 'anuga' as it is not a package.
```

Works fine on Linux. The workaround is importlib-based manual collection:
```python
spec = importlib.util.find_spec('anuga')
anuga_pkg_dir = os.path.dirname(spec.origin)
# then os.walk() to collect all files
```

## rasterio PROJ/GDAL data

Older rasterio had `rasterio.gdal_data()` and `rasterio.proj_data()` functions. Current rasterio (1.4+) stores data directly in the package directory as `rasterio/gdal_data/` and `rasterio/proj_data/` subdirectories.

Must bundle these AND set env vars at runtime via a runtime hook:
```python
os.environ['PROJ_DATA'] = os.path.join(bundle_dir, 'rasterio', 'proj_data')
os.environ['GDAL_DATA'] = os.path.join(bundle_dir, 'rasterio', 'gdal_data')
```

## PROJ version mismatch

pyproj and rasterio may bundle different PROJ versions. rasterio's PROJ data must match its PROJ library. If you set `PROJ_DATA` to pyproj's data, you get:
```
DATABASE.LAYOUT.VERSION.MINOR mismatch
```
Solution: runtime hook prefers rasterio's PROJ data over pyproj's.

## anuga's undeclared dependencies

anuga only declares `numpy` in metadata but requires at runtime:
- six, dill, scipy, mpi4py, netCDF4, matplotlib, triangle, pymetis

All must be in `hiddenimports` in the spec AND explicitly `pip install`ed in CI.

## Building anuga on Windows

anuga uses meson-python with C and Fortran extensions. On Linux, `apt-get install build-essential gfortran` provides everything. On Windows, this is much harder — needs MSVC or MinGW plus gfortran. The `windows-latest` GitHub runner doesn't have gfortran by default.

**Resolution:** Don't compile on Windows CI at all. anuga_core's `build-wheels.yml` builds Windows wheels using conda + `gcc_win-64` (the same toolchain their PyPI workflow uses). run_anuga downloads the pre-built wheel via `gh release download`. See `notes/upstream-wheel-workflow.md`.
