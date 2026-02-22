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
- six, dill, scipy, mpi4py, netCDF4, matplotlib, triangle

All must be in `hiddenimports` in the spec AND explicitly `pip install`ed in CI.

Note: pymetis is NOT needed for single-process runs. The import was made lazy
in `anuga/parallel/distribute_mesh.py` so it only fails when actually used.

## Building anuga on Windows

anuga uses meson-python with C and Fortran extensions. On Linux, `apt-get install build-essential gfortran` provides everything. On Windows, this is much harder — needs MSVC or MinGW plus gfortran. The `windows-latest` GitHub runner doesn't have gfortran by default.

**Resolution:** Don't compile on either platform in run_anuga CI. anuga_core's `build-wheels.yml` builds wheels using conda compilers for both Linux and Windows. run_anuga downloads the pre-built wheels via `gh release download`. See `notes/upstream-wheel-workflow.md`.

## METIS/pymetis cannot compile on Windows

METIS (C library) uses POSIX headers (`sys/resource.h`, `regex.h`) unavailable on Windows. No compiler (GCC, MinGW, MSVC) can build it. The fix was making the pymetis import lazy in anuga_core rather than trying to build it.

## Conda compiler SIGILL issues

Conda's activation scripts set CPU-specific `CFLAGS` (e.g., `-march=nocona` or with SSE/AVX extensions). Wheels built with these flags crash with SIGILL (Illegal Instruction) on machines with older CPUs — including some GitHub Actions runners.

**Resolution:** Override with `-march=x86-64 -mtune=generic` to target baseline x86-64 ISA. See `notes/wheel-build-debug-log.md` for the full debugging story.

Important: `repairwheel`/`auditwheel` do NOT fix instruction set issues — only shared library dependencies.

## Diagnosing native crashes on Windows

When a `.pyd` extension crashes (SIGILL, segfault), Python exits with code 1 and NO traceback. Use `python -X faulthandler` to get a native stack trace. This is critical for debugging SIGILL from conda-compiled extensions.

## Windows CI PATH pollution

`windows-latest` runners have multiple GCC installations on PATH in bash shells:
- Strawberry Perl: `C:\Strawberry\c\bin\`
- MSYS2: `C:\msys64\mingw64\bin\`
- Git for Windows: `C:\Program Files\Git\usr\bin\`

Use `pwsh` shell or explicit PATH filtering when compiler selection matters.
