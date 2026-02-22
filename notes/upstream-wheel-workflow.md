# Upstream PR: build-wheels.yml for anuga_core

## Summary

This document describes the `build-wheels.yml` workflow added to `Hydrata/anuga_core`, intended as a contribution back to `anuga-community/anuga_core`.

## Problem

Downstream projects that depend on anuga (like [run_anuga](https://github.com/Hydrata/run_anuga)) need pre-built wheels to avoid compiling anuga from source in their own CI. This is especially critical on Windows, where:

1. GitHub Actions' `windows-latest` runners don't have Fortran compilers
2. Strawberry Perl's GCC is on the PATH and pollutes meson-python's compiler detection
3. Even with `choco install mingw`, meson-python + Fortran on Windows is fragile

anuga_core already solves this in `python-publish-pypi.yml` using conda's `gcc_win-64` package, but that workflow only runs on PyPI releases. Forks can't publish to PyPI, so they need an alternative way to distribute wheels.

## Solution

A new `build-wheels.yml` workflow that:

1. **Triggers on tag push** (`v*`) and **manual dispatch** (`workflow_dispatch`)
2. Builds wheels for **Linux and Windows**, Python 3.12
3. Uses the **exact same conda + compiler setup** as `python-publish-pypi.yml`:
   - `conda-incubator/setup-miniconda@v3` with Miniforge
   - `gcc_win-64`/`gxx_win-64` on Windows
   - `compilers` on Linux
4. Runs `repairwheel` to produce portable wheels
5. **Creates a GitHub Release** with wheels as assets

## How it complements existing workflows

| Workflow | Trigger | Output | Purpose |
|----------|---------|--------|---------|
| `python-publish-pypi.yml` | GitHub Release published | PyPI packages | Public distribution |
| `build-wheels.yml` | Tag push / manual | GitHub Release assets | CI consumption by downstream projects |

The workflows share the same build approach but differ in distribution method. `build-wheels.yml` is a subset (Python 3.12 only, no macOS, no sdist) since its purpose is narrow: provide installable wheels for CI pipelines.

## Usage by downstream projects

```yaml
# In downstream CI (e.g., run_anuga's release.yml)
- name: Download pre-built anuga wheel
  run: |
    gh release download latest \
      --repo anuga-community/anuga_core \
      --pattern "anuga-*-cp312-cp312-win_amd64.whl" \
      --dir wheels
    pip install wheels/*.whl
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## Extending

To support more Python versions or macOS, add entries to the strategy matrix — the workflow structure supports it, it's just scoped to what's currently needed.
