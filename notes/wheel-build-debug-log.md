# Wheel Build Debug Log

Comprehensive log of all attempts to get the pre-built anuga wheel pipeline working.
This covers the anuga_core wheel build (`build-wheels.yml`) and run_anuga release
(`release.yml`) working together.

## Summary

The goal: build anuga wheels in anuga_core CI (using conda compilers), then have
run_anuga download them to build PyInstaller executables without compiling anuga
from source.

**Total attempts: ~11 workflow runs across both repos over ~2 hours.**

## Attempt 1: Initial wheel build + naive Windows install

**anuga_core**: First `build-wheels.yml` — conda + `gcc_win-64` for Windows,
conda `compilers` for Linux. Both wheels built OK.

**run_anuga**: Windows job downloads wheel, `pip install wheels/*.whl`. Also installs
pymetis from PyPI.

**Result**: Windows FAILED — pymetis compilation fails.
```
error: command 'C:\Strawberry\c\bin\ccache.EXE' failed: No such file or directory
```

**Root cause**: Strawberry Perl's GCC/ccache on PATH interferes with pymetis build.

## Attempt 2: Remove Strawberry Perl from PATH

Added `export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v Strawberry | tr '\n' ':')`.

**Result**: Windows FAILED — different GCC found (MSYS2).
```
fatal error: sys/resource.h: No such file or directory
```

**Root cause**: After removing Strawberry, MSYS2's GCC is found, which also can't
compile METIS (missing POSIX headers).

## Attempt 3: Drop pymetis entirely

Removed pymetis from pip install and hiddenimports.

**Result**: BOTH PLATFORMS FAILED.
```
ModuleNotFoundError: No module named 'pymetis'
```

**Root cause**: anuga hard-imports pymetis at load time via:
`anuga/__init__.py` → `parallel/__init__.py` → `parallel_api.py` →
`sequential_distribute.py` → `distribute_mesh.py` line 46:
`from pymetis import part_graph`

## Attempt 4: Build pymetis wheel alongside anuga

Added pymetis wheel build to anuga_core's build-wheels.yml.

**Result**: FAILED — pybind11 (pymetis build dep) fails to build.

**Root cause**: Deep dependency chain: pymetis → pybind11 → cmake. The cmake
build fails in the conda GCC environment.

## Attempt 5: Use --no-isolation for pymetis build

Pre-install pybind11 and build pymetis with `pip wheel --no-isolation`.

**Result**: FAILED — Strawberry Perl GCC found again in base conda env.

## Attempt 6: Remove Strawberry from PATH in pymetis build

**Result**: FAILED — MSYS2's GCC found instead. Same `sys/resource.h` error.

## Attempt 7: Use MSVC compiler (ilammy/msvc-dev-cmd)

Used `ilammy/msvc-dev-cmd@v1` action with `CC=cl CXX=cl`.

**Result**: FAILED — meson found `C:\Program Files\Git\usr\bin\link.EXE` (GNU link)
instead of MSVC's `link.exe`. Bash shell adds Git usr/bin to PATH.

## Attempt 8: Use pwsh shell for MSVC build

Switched to `shell: pwsh` to avoid bash PATH pollution.

**Result**: FAILED — MSVC compiles OK but METIS C code needs `regex.h` (POSIX header).
```
metis/libmetis/gklib_defs.h(24): fatal error C1083: 'regex.h': No such file or directory
```

**Key insight: METIS does NOT compile on Windows with ANY compiler.**
- GCC needs `sys/resource.h` (POSIX)
- MSVC needs `regex.h` (POSIX)

## Attempt 9: Make pymetis import lazy in anuga_core ✅

Instead of trying to build pymetis, we made the import lazy in anuga_core's
`anuga/parallel/distribute_mesh.py`:

```python
# Before (hard import at module load):
from pymetis import part_graph

# After (lazy import):
try:
    from pymetis import part_graph
except ImportError:
    part_graph = None
```

With guard at the actual usage site (line ~206):
```python
if part_graph is None:
    raise ImportError("pymetis is required for parallel mesh distribution.")
```

**Result**: Wheel build succeeded! run_anuga Windows passed all smoke tests!
But Linux FAILED with SIGILL.

## Attempt 10: Linux SIGILL from conda-built wheel

**Result**: Linux FAILED — `Illegal instruction (core dumped)` when importing anuga.

**Root cause**: The manylinux wheel built with conda compilers used CPU-specific
instructions (likely AVX2+ from `-march=nocona` or similar in conda's CFLAGS
activation scripts). The GitHub Actions ubuntu-latest runner had an older CPU.

**Temporary fix**: Reverted Linux to source build from git (the original approach
that always worked). Windows stayed on pre-built wheel.

## Attempt 11: Windows SIGILL too! (flaky — first run passed)

After the Linux revert, Windows started failing with exit code 1 and NO output.

**Diagnosis**: Added `-X faulthandler` flag to Python. Revealed:
```
Windows fatal exception: code 0xc000001d
Fatal Python error: Illegal instruction
Extension modules: ..., anuga.geometry.polygon_ext (total: 4)
```

**Root cause**: Same as Linux! The conda-compiled `.pyd` files used CPU instructions
not available on the GitHub Actions runner. The first Windows run got lucky with a
runner that had the right CPU features; subsequent runs got different hardware.

## Attempt 12: Baseline x86-64 compilation ✅✅

Added to both Linux and Windows build steps in `build-wheels.yml`:
```bash
export CFLAGS="-march=x86-64 -mtune=generic"
export CXXFLAGS="-march=x86-64 -mtune=generic"
export FFLAGS="-march=x86-64 -mtune=generic"
```

This overrides conda's aggressive optimization flags and targets baseline x86-64
ISA (SSE2 only), ensuring the wheel runs on ANY x86-64 machine.

**Result**: Both Windows AND Linux passed all smoke tests!
- anuga imports successfully
- PyInstaller builds single-file executable
- `validate`, `info` commands work
- Full simulation runs and produces .sww + .tif output files

## Key Lessons

### 1. Conda compilers optimize for the build machine's CPU
Conda's activation scripts set `CFLAGS` with architecture-specific flags. The
resulting binaries may not run on machines with older CPUs. Always override with
`-march=x86-64 -mtune=generic` for portable wheels.

### 2. SIGILL can be silent without faulthandler
On Windows, a SIGILL crash in a `.pyd` extension produces exit code 1 with NO
Python traceback. Always use `python -X faulthandler` to diagnose native crashes.

### 3. METIS cannot compile on Windows
METIS (the graph partitioning library that pymetis wraps) uses POSIX headers
(`sys/resource.h`, `regex.h`) that aren't available on Windows with any compiler.
The solution is to make the import lazy in the Python code that uses it.

### 4. Windows CI runners have multiple GCC installations
`windows-latest` runners have:
- Strawberry Perl's GCC (`C:\Strawberry\c\bin\`)
- MSYS2's GCC (`C:\msys64\mingw64\bin\`)
- Git for Windows' utilities (`C:\Program Files\Git\usr\bin\`)
All pollute the PATH in bash shells. Use `pwsh` shell or explicit PATH filtering.

### 5. "Flaky" tests may be CPU-dependent
A test that passes on one run and fails on the next with SIGILL is NOT random —
it's getting different hardware. GitHub Actions runners are assigned from a pool
with varying CPU capabilities.

### 6. repairwheel doesn't fix instruction set issues
`repairwheel` (and `auditwheel`) only handle shared library dependencies. They
do NOT fix CPU instruction set incompatibilities. That must be fixed at compile time.

## Final Working Configuration

### anuga_core/build-wheels.yml
- Trigger: tag push (`v*`) + workflow_dispatch
- Linux: conda `compilers`, CFLAGS=`-march=x86-64 -mtune=generic`
- Windows: conda `gcc_win-64`/`gxx_win-64`, same CFLAGS
- Both: `python -m build --wheel` → `repairwheel`
- Release job creates GitHub Release with wheel assets

### anuga_core/anuga/parallel/distribute_mesh.py
- pymetis import made lazy (try/except at module level)
- Guard at actual usage site raises ImportError with helpful message

### run_anuga/release.yml
- Both platforms: download pre-built wheel via `gh release download`
- Windows: `mpi4py/setup-mpi` for MSMPI
- Linux: `apt-get install libopenmpi-dev openmpi-bin`
- Smoke tests: validate, info, run simulation, verify output files

### Successful run: 22280082450
- Linux: ✅ All passed
- Windows: ✅ All passed
- Wheel release: v3.2.0-wheels.9
