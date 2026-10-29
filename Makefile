# `make check` is the static gate, and CI's `lint` job runs it (TASK-4017, epic 3995).
# Needs: pip install ruff==0.15.2   (CI pins the same version; locally: make check RUFF="uvx ruff@0.15.2")
# The unit tests are `make test` (CI's `test` job runs the same pytest line, with coverage).
RUFF ?= ruff
PY ?= python

.PHONY: check lint marker-lint test

check: lint marker-lint

# --isolated pins the exact rule set (fleet standard D3: E4,E7,E9,F,I) regardless of pyproject.toml. It runs from
# outside the repo on its absolute path, exactly like the epic's acceptance command: ruff's isort groups `run_anuga`
# imports differently when the cwd is the repo root, and this tree is sorted for the from-outside grouping.
# A superset of what CI ran before (default rules over run_anuga/ + tests/).
lint:
	cd / && $(RUFF) check --isolated --no-cache --select E4,E7,E9,F,I $(CURDIR)

# Fleet marker/quarantine taxonomy lint: static scan, no deps. Cap = the current conditional-skip baseline.
marker-lint:
	$(PY) .github/scripts/marker_lint.py --repo . --check --cap conditional-skip=29

test:
	RUN_ANUGA_EXPECT_RASTERIO=1 pytest tests/ -m "not requires_anuga and not requires_geo and not mpi"
