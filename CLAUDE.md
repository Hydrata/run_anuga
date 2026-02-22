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

- `notes/v0.1.0-release-status.md` — Current release status, what works, what's broken, next steps
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

## Release Process

1. Ensure CI green on both `Hydrata/anuga_core` and `Hydrata/run_anuga`
2. Tag: `git tag v0.x.x -m "message" && git push origin v0.x.x`
3. Release workflow builds single-file executables for Windows + Linux
4. Smoke tests run full simulation on both platforms before creating release
5. Release assets: `run-anuga-windows-amd64.zip` (exe + examples), `run-anuga-linux-amd64.tar.gz`

## Current Blocker

Windows build fails because anuga (meson-python with Fortran extensions) won't compile on Windows CI. See `notes/v0.1.0-release-status.md` for details and options.
