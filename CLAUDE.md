# run_anuga

CLI tool for running ANUGA flood simulations from Hydrata scenario packages.

## Repo Layout

```
run_anuga/          # Python package (cli.py, run.py, config.py, run_utils.py, etc.)
examples/small_test/  # Small 200x200m test scenario
tests/              # pytest unit tests (56 tests, no ANUGA required)
test-docker/        # Docker-based end-to-end tests (phase1/2/3.sh)
pyinstaller_hooks/  # Runtime hooks for PyInstaller builds
notes/              # Session notes and lessons learned
```

## Key Files

- `run_anuga.spec` — PyInstaller spec (single-file exe mode)
- `.github/workflows/release.yml` — Builds Windows + Linux executables on tag push
- `.github/workflows/ci.yml` — Unit tests + lint on push/PR
- `run_anuga/cli.py` — CLI entry point, accepts scenario.json or directory
- `run_anuga/run_utils.py` — Core simulation utilities (mesh, boundaries, post-processing)
- `run_anuga/_imports.py` — Lazy import helper with helpful error messages

## Session Notes

- `notes/v0.1.0-release-status.md` — Release history and current status
- `notes/pyinstaller-lessons.md` — PyInstaller gotchas (meson-python, PROJ data, onefile vs onedir)

## Development

```bash
# Install dev deps
pip install -e ".[dev]"

# Run tests (no ANUGA needed)
pytest tests/ -v --ignore=tests/test_integration.py

# Lint
ruff check run_anuga/ tests/

# Docker end-to-end tests (13 tests)
bash test-docker/test_readme.sh
```

## Release Process (Two-Step)

### Step 1: Build anuga wheels (in `Hydrata/anuga_core`)
1. Tag anuga_core: `git tag v3.x.x -m "message" && git push origin v3.x.x`
2. `build-wheels.yml` builds Linux + Windows wheels (Python 3.12, conda + gcc_win-64)
3. Creates a GitHub Release with wheels as assets

### Step 2: Build run_anuga executables (in `Hydrata/run_anuga`)
1. Ensure anuga_core has a release with wheels
2. Tag run_anuga: `git tag v0.x.x -m "message" && git push origin v0.x.x`
3. `release.yml` downloads pre-built wheels from anuga_core's latest release
4. Builds single-file executables for Windows + Linux
5. Smoke tests run full simulation on both platforms
6. Creates release with assets: `run-anuga-windows.zip`, `run-anuga-ubuntu2204.tar.gz`, `run-anuga-ubuntu2404.tar.gz`

To target a specific anuga_core release: trigger `release.yml` via `workflow_dispatch` with `anuga_core_tag` input.
