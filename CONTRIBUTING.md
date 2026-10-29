# Contributing to run_anuga

run_anuga is the ANUGA flood-simulation runner behind Hydrata. It is installed editable on Hydrata's servers and baked into
its AWS Batch images, so a change here can change what a production simulation computes. Keep changes small and tested.

Do not report security problems in an issue or pull request. Email david.kennewell@hydrata.com instead.

## Set up

Python 3.10 or newer (CI runs 3.10 and 3.12).

```bash
git clone https://github.com/Hydrata/run_anuga
cd run_anuga
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

`[dev]` is the light set: pytest, ruff, rasterio, zarr and friends. It does not build the ANUGA engine. The heavier
extras need system libraries and are only for the tests that say so:

- `[sim]` adds the GDAL bindings and geopandas (needs `libgdal-dev`; pin `GDAL==$(gdal-config --version)`).
- `[full]` adds the ANUGA engine itself (compiled; CI installs the Hydrata `anuga_core` fork at a pinned commit).

## Tests

The tests that run on every pull request, without GDAL or the ANUGA engine:

```bash
RUN_ANUGA_EXPECT_RASTERIO=1 pytest tests/ -m "not requires_anuga and not requires_geo and not mpi"
```

Tests that need more are marked, and CI runs them in their own jobs:

| Marker | Needs | CI command |
|--------|-------|------------|
| `requires_geo` | GDAL bindings (`[sim]`) | `pytest tests/ -m "requires_geo and not requires_anuga"` |
| `requires_anuga` | the ANUGA engine (`[full]` + the fork); includes the Merewether field benchmark (about 9 minutes) | `pytest tests/ -m requires_anuga --timeout=1800` |
| `mpi` | MPI, not run in CI | |

A test that skips or xfails must say why: `.github/scripts/marker_lint.py` rejects bare skips and bare xfails.

## `make check`

`make check` is the static gate, and CI's `lint` job runs it:

```bash
pip install ruff==0.15.2     # CI pins this exact version; or: make check RUFF="uvx ruff@0.15.2"
make check
```

It runs `make lint` (ruff with the fleet rule set E4, E7, E9, F, I over the whole repo; the tree is clean, so there is
no ignore list) and `make marker-lint` (the skip/xfail taxonomy lint described above). The lint step runs from outside the
repo on its absolute path, because ruff groups `run_anuga` imports differently when the working directory is the repo
root. The unit tests are `make test`, the same pytest line as above.

Bump ruff deliberately, together with a fix sweep in its own commit.

## Pull requests

- Branch from `main`; open the pull request against `main`. One logical change per pull request.
- Add or update a test with every behaviour change, and show it fails without the fix.
- Run `make check` and the unit tests before pushing.
- Do not commit simulation outputs, logs, keys, or customer data.
- Changes under `.github/` are reviewed by the code owners (see `.github/CODEOWNERS`).
