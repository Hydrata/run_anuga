# run_anuga Test Suite Design

## Current State

| Layer | Tests | Coverage |
|-------|-------|----------|
| Unit tests (`tests/`) | 56 across 6 files | config, callbacks, imports, defaults, logging, cli |
| Integration (`test_integration.py`) | 1 parametrized | Full sim, excluded from CI |
| Docker E2E (`test-docker/`) | 13 assertions in 3 phases | Install, run, post-process |
| **Total** | **70** | |

### Key Gaps
- `run_utils.py` (1125 lines, largest module) — **0 unit tests**
- `run.py` (327 lines, simulation orchestrator) — **0 unit tests**
- MPI parallel execution — **0 tests anywhere**
- Windows platform — **not in CI matrix**
- Component-level tests (with geo deps but without ANUGA) — **no layer exists**
- No `conftest.py` — no shared fixtures, no auto-skip for missing deps

---

## Test Pyramid Design

```
           /\
          /  \   E2E: Docker tests + PyInstaller smoke tests
         /    \  (7 tests) — install, run, post-process, verify outputs
        /------\
       /        \   Integration: Full ANUGA simulation
      /          \  (12 tests) — run_sim, MPI parallel, checkpoint restart
     /------------\
    /              \   Component: Geo deps (shapely/rasterio) but no ANUGA
   /                \  (30+ tests) — boundaries, mesh regions, frictions, raster ops
  /------------------\
 /                    \   Unit: Pure logic, no heavy deps
/                      \  (80+ tests) — config, callbacks, cli, imports, defaults,
/________________________\ logging, yieldstep calc, package loading, data transforms
```

---

## Layer 1: Unit Tests (No Heavy Dependencies)

Target: **80+ tests**, run in CI on all platforms, < 10s total.

### 1.1 Existing (keep as-is)
- `test_config.py` (14 tests) — Pydantic model validation
- `test_callbacks.py` (15 tests) — callback protocol + implementations
- `test_cli.py` (8 tests) — CLI via subprocess
- `test_imports.py` (8 tests) — lazy import helper
- `test_defaults.py` (10 tests) — constants validation
- `test_logging_setup.py` (17 tests) — logging configuration

### 1.2 New: `test_cli_resolve.py` — resolve_package_dir edge cases
Tests for `cli.py:resolve_package_dir()`:

```python
def test_resolve_with_scenario_json_file(tmp_path):
    (tmp_path / "scenario.json").write_text("{}")
    assert resolve_package_dir(str(tmp_path / "scenario.json")) == str(tmp_path)

def test_resolve_with_directory(tmp_path):
    assert resolve_package_dir(str(tmp_path)) == str(tmp_path)

def test_resolve_rejects_non_scenario_file(tmp_path):
    (tmp_path / "other.json").write_text("{}")
    with pytest.raises(argparse.ArgumentTypeError):
        resolve_package_dir(str(tmp_path / "other.json"))

def test_resolve_nonexistent_path():
    with pytest.raises(argparse.ArgumentTypeError):
        resolve_package_dir("/nonexistent/path")

def test_resolve_relative_path(tmp_path, monkeypatch):
    (tmp_path / "scenario.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    result = resolve_package_dir(".")
    assert os.path.isabs(result)
```

### 1.3 New: `test_package_loading.py` — _load_package_data logic
Tests for `run_utils.py:_load_package_data()` — this function only needs JSON files and pydantic, not ANUGA.

```python
def test_load_minimal_package(scenario_package):
    """Package with just scenario.json + boundary loads successfully."""
    data = _load_package_data(str(scenario_package))
    assert "scenario_config" in data
    assert "run_label" in data
    assert "output_directory" in data
    assert "boundary" in data

def test_load_creates_output_directory(scenario_package):
    data = _load_package_data(str(scenario_package))
    assert os.path.isdir(data["output_directory"])

def test_load_creates_checkpoint_directory(scenario_package):
    data = _load_package_data(str(scenario_package))
    assert os.path.isdir(data["checkpoint_directory"])

def test_load_missing_scenario_json(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_package_data(str(tmp_path))

def test_load_with_optional_inputs(scenario_package_full):
    """Package with friction, inflow, structure, mesh_region."""
    data = _load_package_data(str(scenario_package_full))
    assert "friction" in data
    assert "inflow" in data

def test_run_label_format(scenario_package):
    data = _load_package_data(str(scenario_package))
    assert data["run_label"] == "run_1_1_1"

def test_mesh_filepath_extension(scenario_package):
    data = _load_package_data(str(scenario_package))
    assert data["mesh_filepath"].endswith(".msh")
```

### 1.4 New: `test_data_transforms.py` — pure-logic functions from run_utils
Tests for functions that transform GeoJSON dicts without needing geo libraries:

```python
# make_new_inflow — creates GeoJSON feature dict
def test_make_new_inflow_structure():
    result = make_new_inflow("test_1", [[0, 0], [1, 1]], 0.5)
    assert result["type"] == "Feature"
    assert result["geometry"]["type"] == "LineString"
    assert result["properties"]["flow"] == 0.5

def test_make_new_inflow_id():
    result = make_new_inflow("inflow_42", [[0, 0], [1, 1]], 1.0)
    assert result["properties"]["id"] == "inflow_42"

# lookup_boundary_tag — finds which boundary contains a point index
def test_lookup_boundary_tag_found():
    tags = {"ocean": [0, 1, 2], "river": [3, 4]}
    assert lookup_boundary_tag(1, tags) == "ocean"
    assert lookup_boundary_tag(4, tags) == "river"

def test_lookup_boundary_tag_not_found():
    tags = {"ocean": [0, 1]}
    assert lookup_boundary_tag(99, tags) is None

# is_dir_check — argparse type validator
def test_is_dir_check_valid(tmp_path):
    assert is_dir_check(str(tmp_path)) == str(tmp_path)

def test_is_dir_check_file(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(argparse.ArgumentTypeError):
        is_dir_check(str(f))

def test_is_dir_check_nonexistent():
    with pytest.raises(argparse.ArgumentTypeError):
        is_dir_check("/nonexistent")
```

### 1.5 New: `test_yieldstep.py` — extracted yieldstep calculation
**Requires extracting `compute_yieldstep()` from `run.py` lines 244-251.**

```python
from run_anuga.defaults import MAX_YIELDSTEPS, MIN_YIELDSTEP_S, MAX_YIELDSTEP_S

def test_yieldstep_short_duration():
    """Duration < MAX_YIELDSTEPS * MIN_YIELDSTEP_S → clamped to MIN."""
    assert compute_yieldstep(60) == MIN_YIELDSTEP_S  # 60s / 100 = 0.6 → min 60

def test_yieldstep_medium_duration():
    """Duration that falls within bounds."""
    result = compute_yieldstep(30000)  # 30000 / 100 = 300
    assert MIN_YIELDSTEP_S <= result <= MAX_YIELDSTEP_S

def test_yieldstep_long_duration():
    """Duration > MAX_YIELDSTEPS * MAX_YIELDSTEP_S → clamped to MAX."""
    assert compute_yieldstep(1_000_000) == MAX_YIELDSTEP_S

def test_yieldstep_exact_boundary():
    """Duration = MAX_YIELDSTEPS * MIN_YIELDSTEP_S."""
    assert compute_yieldstep(MAX_YIELDSTEPS * MIN_YIELDSTEP_S) == MIN_YIELDSTEP_S

# Property-based
@given(duration=st.integers(min_value=1, max_value=10_000_000))
def test_yieldstep_always_in_bounds(duration):
    result = compute_yieldstep(duration)
    assert MIN_YIELDSTEP_S <= result <= MAX_YIELDSTEP_S
```

### 1.6 New: `test_callbacks_http.py` — HydrataCallback error handling
```python
@patch("run_anuga.callbacks.import_optional")
def test_hydrata_callback_http_error_logged(mock_import, caplog):
    """HTTP errors are logged but not raised."""
    mock_requests = MagicMock()
    mock_response = MagicMock(status_code=500, text="Internal Server Error")
    mock_requests.patch.return_value = mock_response
    mock_import.return_value = mock_requests

    cb = HydrataCallback("user", "pass", "http://example.com", 1, 1, 1)
    cb.on_status("running")
    assert "500" in caplog.text or "error" in caplog.text.lower()

@patch("run_anuga.callbacks.import_optional")
def test_hydrata_callback_on_file_missing(mock_import):
    """on_file with nonexistent file raises."""
    cb = HydrataCallback("user", "pass", "http://example.com", 1, 1, 1)
    with pytest.raises(FileNotFoundError):
        cb.on_file("result", "/nonexistent/file.tif")
```

### 1.7 New: Snapshot tests for CLI output
Using `syrupy` (add to dev deps):

```python
def test_help_output_snapshot(snapshot):
    result = subprocess.run(
        [sys.executable, "-m", "run_anuga.cli", "--help"],
        capture_output=True, text=True,
    )
    assert result.stdout == snapshot

def test_validate_output_snapshot(snapshot):
    result = subprocess.run(
        [sys.executable, "-m", "run_anuga.cli", "validate", FIXTURE_DIR],
        capture_output=True, text=True,
    )
    assert result.stdout == snapshot

def test_info_output_snapshot(snapshot):
    result = subprocess.run(
        [sys.executable, "-m", "run_anuga.cli", "info", FIXTURE_DIR],
        capture_output=True, text=True,
    )
    assert result.stdout == snapshot
```

---

## Layer 2: Component Tests (Geo Deps, No ANUGA)

Target: **30+ tests**, run in CI with `[sim]` extra or auto-skipped.
Marker: `@pytest.mark.requires_geo`

### 2.1 New: `test_boundaries.py` — boundary polygon creation
Tests for `run_utils.py:create_boundary_polygon_from_boundaries()`:

```python
@pytest.mark.requires_geo
class TestCreateBoundaryPolygon:
    def test_simple_rectangle(self):
        """4 boundary segments forming a rectangle → closed polygon."""
        geojson = make_boundary_geojson([
            {"coords": [[0,0],[100,0]], "name": "south", "type": "external"},
            {"coords": [[100,0],[100,100]], "name": "east", "type": "external"},
            {"coords": [[100,100],[0,100]], "name": "north", "type": "external"},
            {"coords": [[0,100],[0,0]], "name": "west", "type": "external"},
        ])
        polygon, tags = create_boundary_polygon_from_boundaries(geojson)
        assert len(polygon) >= 4
        assert set(tags.keys()) == {"south", "east", "north", "west"}

    def test_clockwise_ordering(self):
        """Boundaries are sorted clockwise regardless of input order."""
        # Provide boundaries in random order
        geojson = make_boundary_geojson([...])  # shuffled
        polygon, tags = create_boundary_polygon_from_boundaries(geojson)
        # Verify clockwise via signed area
        assert signed_area(polygon) < 0  # clockwise

    def test_internal_boundaries_excluded(self):
        """Internal boundaries filtered out."""
        geojson = make_boundary_geojson([
            {"coords": [[0,0],[100,0]], "name": "south", "type": "external"},
            {"coords": [[50,50],[60,60]], "name": "wall", "type": "internal"},
            ...
        ])
        polygon, tags = create_boundary_polygon_from_boundaries(geojson)
        assert "wall" not in tags

    def test_no_external_boundaries_raises(self):
        """Empty external boundaries → error."""
        geojson = {"type": "FeatureCollection", "features": []}
        with pytest.raises((AttributeError, IndexError)):
            create_boundary_polygon_from_boundaries(geojson)
```

### 2.2 New: `test_mesh_regions.py` — mesh region extraction
```python
@pytest.mark.requires_geo
def test_make_interior_regions_basic():
    input_data = {
        "mesh_region": {
            "features": [
                {"geometry": {"coordinates": [[[0,0],[1,0],[1,1],[0,1],[0,0]]]},
                 "properties": {"resolution": 5.0}}
            ]
        }
    }
    regions = make_interior_regions(input_data)
    assert len(regions) == 1
    polygon, resolution = regions[0]
    assert resolution == 5.0

@pytest.mark.requires_geo
def test_make_interior_regions_no_mesh_region():
    input_data = {}
    regions = make_interior_regions(input_data)
    assert regions == []

@pytest.mark.requires_geo
def test_make_frictions_with_buildings():
    input_data = {
        "structure": {
            "features": [
                {"geometry": {"coordinates": [[[0,0],[1,0],[1,1],[0,1],[0,0]]]},
                 "properties": {"method": "Mannings"}}
            ]
        }
    }
    frictions = make_frictions(input_data)
    assert any(f[1] == defaults.BUILDING_MANNINGS_N for f in frictions)
    assert any(f[1] == defaults.DEFAULT_MANNINGS_N for f in frictions)  # 'All' default

@pytest.mark.requires_geo
def test_make_interior_holes_and_tags():
    input_data = {
        "structure": {
            "features": [
                {"geometry": {"coordinates": [[[0,0],[1,0],[1,1],[0,1],[0,0]]]},
                 "properties": {"method": "Hole", "name": "building_1"}}
            ]
        }
    }
    holes, tags = make_interior_holes_and_tags(input_data)
    assert holes is not None
    assert len(holes) == 1
    assert tags[0] == "building_1"
```

### 2.3 New: `test_coordinate_checks.py` — geometry validation
```python
@pytest.mark.requires_geo
def test_point_inside_polygon():
    polygon = [[0,0], [10,0], [10,10], [0,10]]
    assert check_coordinates_are_in_polygon([[5, 5]], polygon) is True

@pytest.mark.requires_geo
def test_point_outside_polygon():
    polygon = [[0,0], [10,0], [10,10], [0,10]]
    assert check_coordinates_are_in_polygon([[15, 15]], polygon) is False

@pytest.mark.requires_geo
def test_single_point_not_nested():
    """Single point [x, y] instead of [[x, y]]."""
    polygon = [[0,0], [10,0], [10,10], [0,10]]
    assert check_coordinates_are_in_polygon([5, 5], polygon) is True

@pytest.mark.requires_geo
def test_multiple_points_all_inside():
    polygon = [[0,0], [10,0], [10,10], [0,10]]
    coords = [[1,1], [5,5], [9,9]]
    assert check_coordinates_are_in_polygon(coords, polygon) is True

@pytest.mark.requires_geo
def test_multiple_points_one_outside():
    polygon = [[0,0], [10,0], [10,10], [0,10]]
    coords = [[1,1], [15,15]]
    assert check_coordinates_are_in_polygon(coords, polygon) is False

# Property-based
@pytest.mark.requires_geo
@given(
    x=st.floats(min_value=0.01, max_value=9.99),
    y=st.floats(min_value=0.01, max_value=9.99),
)
def test_interior_points_always_inside(x, y):
    polygon = [[0,0], [10,0], [10,10], [0,10]]
    assert check_coordinates_are_in_polygon([[x, y]], polygon) is True
```

### 2.4 New: `test_polar_quadrants.py` — angle correction
```python
def test_correction_first_quadrant():
    # Positive x, positive y → 0
    assert correction_for_polar_quadrants(1.0, 1.0) == 0

def test_correction_second_quadrant():
    # Negative x, positive y → pi
    assert correction_for_polar_quadrants(-1.0, 1.0) == pytest.approx(math.pi)

def test_correction_third_quadrant():
    # Negative x, negative y → pi
    assert correction_for_polar_quadrants(-1.0, -1.0) == pytest.approx(math.pi)

def test_correction_fourth_quadrant():
    # Positive x, negative y → 2*pi
    assert correction_for_polar_quadrants(1.0, -1.0) == pytest.approx(2 * math.pi)

# Property-based
@given(
    base=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    height=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_correction_always_in_valid_range(base, height):
    assume(base != 0 or height != 0)
    result = correction_for_polar_quadrants(base, height)
    assert 0 <= result <= 2 * math.pi
```

### 2.5 New: `test_raster_operations.py` — raster processing (requires rasterio)
```python
@pytest.mark.requires_geo
def test_clip_and_resample(small_geotiff, tmp_path):
    """Clip raster to cutline and resample."""
    cutline = create_test_shapefile(tmp_path / "cutline.shp", ...)
    dst = str(tmp_path / "clipped.tif")
    _clip_and_resample(str(small_geotiff), dst, str(cutline), resolution=5.0)

    with rasterio.open(dst) as ds:
        assert ds.width > 0
        assert ds.height > 0
        assert ds.crs is not None

@pytest.mark.requires_geo
def test_burn_structures_into_raster(small_geotiff, tmp_path):
    """Burn building polygons into raster."""
    structures = tmp_path / "structures.geojson"
    structures.write_text(json.dumps({...}))

    import shutil
    raster = tmp_path / "dem.tif"
    shutil.copy(str(small_geotiff), str(raster))

    result = burn_structures_into_raster(str(structures), str(raster))
    assert result is True
    assert (tmp_path / "dem_original.tif").exists()  # backup created
```

---

## Layer 3: Integration Tests (Requires ANUGA)

Target: **12+ tests**, run manually or in dedicated CI job.
Marker: `@pytest.mark.requires_anuga`

### 3.1 Existing: `test_integration.py` — keep and expand
Current: 1 parametrized test checking output files.

### 3.2 New: `test_sim_lifecycle.py` — simulation phases
```python
@pytest.mark.requires_anuga
@pytest.mark.slow
class TestSimLifecycle:
    def test_run_sim_produces_sww(self, small_test_copy):
        """Full simulation produces .sww file."""
        run_sim(str(small_test_copy))
        sww_files = list(small_test_copy.glob("outputs_*/*.sww"))
        assert len(sww_files) >= 1

    def test_run_sim_produces_geotiffs(self, small_test_copy):
        """Full simulation produces depth + velocity GeoTIFFs."""
        run_sim(str(small_test_copy))
        tif_files = list(small_test_copy.glob("outputs_*/*_max.tif"))
        assert len(tif_files) >= 2
        names = {f.stem for f in tif_files}
        assert any("depth" in n for n in names)
        assert any("velocity" in n for n in names)

    def test_run_sim_log_file(self, small_test_copy):
        """Simulation creates log file."""
        run_sim(str(small_test_copy))
        log_files = list(small_test_copy.glob("outputs_*/run_anuga_*.log"))
        assert len(log_files) == 1
        content = log_files[0].read_text()
        assert "100%" in content  # completion logged

    def test_run_sim_with_callback(self, small_test_copy):
        """Callback receives status updates during simulation."""
        statuses = []
        class RecordingCallback:
            def on_status(self, status, **kw): statuses.append(status)
            def on_metric(self, key, value): pass
            def on_file(self, key, filepath): pass

        run_sim(str(small_test_copy), callback=RecordingCallback())
        assert any("%" in s for s in statuses)
        assert "complete" in statuses[-1].lower() or "100" in statuses[-1]

    def test_run_sim_sww_valid(self, small_test_copy):
        """SWW file is valid NetCDF with expected variables."""
        run_sim(str(small_test_copy))
        sww = list(small_test_copy.glob("outputs_*/*.sww"))[0]
        import netCDF4
        ds = netCDF4.Dataset(str(sww))
        assert "stage" in ds.variables
        assert "xmomentum" in ds.variables
        assert "ymomentum" in ds.variables
        ds.close()
```

### 3.3 New: `test_post_process.py` — GeoTIFF generation from SWW
```python
@pytest.mark.requires_anuga
@pytest.mark.slow
class TestPostProcess:
    @pytest.fixture(autouse=True)
    def run_simulation(self, small_test_copy):
        """Run simulation once, then test post-processing."""
        run_sim(str(small_test_copy))
        self.package_dir = small_test_copy

    def test_post_process_creates_tiffs(self):
        post_process_sww(str(self.package_dir))
        tifs = list(self.package_dir.glob("outputs_*/*_max.tif"))
        assert len(tifs) >= 2

    def test_post_process_tiff_has_valid_crs(self):
        post_process_sww(str(self.package_dir))
        tifs = list(self.package_dir.glob("outputs_*/*depth_max.tif"))
        import rasterio
        with rasterio.open(str(tifs[0])) as ds:
            assert ds.crs is not None

    def test_post_process_custom_resolution(self):
        post_process_sww(str(self.package_dir), output_raster_resolution=10.0)
        tifs = list(self.package_dir.glob("outputs_*/*depth_max.tif"))
        import rasterio
        with rasterio.open(str(tifs[0])) as ds:
            assert abs(ds.res[0] - 10.0) < 0.1

    def test_post_process_idempotent(self):
        """Running post-process twice doesn't error."""
        post_process_sww(str(self.package_dir))
        post_process_sww(str(self.package_dir))  # should overwrite cleanly
```

### 3.4 New: `test_checkpoint.py` — checkpoint/restart
```python
@pytest.mark.requires_anuga
@pytest.mark.slow
class TestCheckpoint:
    def test_checkpoint_files_created(self, small_test_copy):
        """Batch 1 creates checkpoint pickle files."""
        run_sim(str(small_test_copy), batch_number=1)
        pickles = list(small_test_copy.glob("**/*.pickle"))
        assert len(pickles) >= 1

    def test_checkpoint_restart(self, small_test_copy):
        """Batch 2 resumes from batch 1 checkpoint."""
        run_sim(str(small_test_copy), batch_number=1)
        # Find checkpoint time from pickle filename
        pickles = list(small_test_copy.glob("**/*.pickle"))
        # Extract time from filename pattern
        checkpoint_time = extract_checkpoint_time(pickles[0].name)
        run_sim(str(small_test_copy), batch_number=2, checkpoint_time=checkpoint_time)
        # Should produce output for batch 2
        logs = list(small_test_copy.glob("outputs_*/run_anuga_2.log"))
        assert len(logs) == 1
```

---

## Layer 4: MPI Parallel Tests

Target: **6+ tests**, run with `mpirun -np 2`.
Marker: `@pytest.mark.requires_anuga` + `@pytest.mark.mpi`

### 4.1 New: `test_mpi.py` — parallel execution
These tests must be run via `mpirun -np 2 python -m pytest tests/test_mpi.py`

```python
@pytest.mark.requires_anuga
@pytest.mark.mpi
class TestMPIParallel:
    def test_parallel_run_produces_output(self, small_test_copy):
        """2-process MPI run produces merged SWW."""
        run_sim(str(small_test_copy))
        sww = list(small_test_copy.glob("outputs_*/*.sww"))
        assert len(sww) >= 1

    def test_parallel_produces_same_tiffs_as_serial(self, small_test_copy, serial_reference):
        """Parallel output is consistent with serial output."""
        run_sim(str(small_test_copy))
        # Compare depth_max.tif between serial and parallel
        parallel_tif = list(small_test_copy.glob("outputs_*/*depth_max.tif"))[0]
        assert_geotiff_similar(serial_reference, parallel_tif, tolerance=0.01)

    def test_parallel_checkpoint_restart(self, small_test_copy):
        """Checkpoint from 2-process run restores correctly."""
        run_sim(str(small_test_copy), batch_number=1)
        pickles = list(small_test_copy.glob("**/*.pickle"))
        # Each rank produces its own pickle
        assert len(pickles) >= 2  # one per process
```

### 4.2 MPI unit tests (mockable, no real MPI needed)
```python
class TestMPICheckpointSync:
    """Test checkpoint synchronization logic with mocked MPI."""

    @patch("run_anuga.run.barrier")
    @patch("run_anuga.run.send")
    @patch("run_anuga.run.receive")
    def test_all_ranks_find_checkpoint(self, mock_recv, mock_send, mock_barrier):
        """When all ranks find their pickle, sync succeeds."""
        # Test the checkpoint sync loop logic
        ...

    @patch("run_anuga.run.barrier")
    def test_checkpoint_sync_timeout(self, mock_barrier):
        """When checkpoint file missing after retries, raises."""
        ...
```

### 4.3 Running MPI tests in CI
```yaml
# .github/workflows/ci.yml addition
  test-mpi:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: |
          sudo apt-get install -y libopenmpi-dev openmpi-bin
          pip install -e ".[sim]" mpi4py
          pip install anuga@...  # from pre-built wheel
      - run: |
          mpirun -np 2 --allow-run-as-root \
            python -m pytest tests/test_mpi.py -v
```

---

## Layer 5: E2E Tests (Docker + PyInstaller)

Target: **15+ assertions**, run on-demand or scheduled.

### 5.1 Existing Docker tests — keep and extend
Current phases: install (4), sim (7), post-process (2)

### 5.2 New: `phase4.sh` — MPI parallel run
```bash
#!/usr/bin/env bash
source "$(dirname "$0")/helpers.sh"

# Phase 4: MPI Parallel Simulation
test_step 1 "Run simulation with 2 MPI processes" \
    "mpirun -np 2 --allow-run-as-root run-anuga run /tmp/workdir/small_test/"

test_step 2 "Verify output files from parallel run" \
    "test -f /tmp/workdir/small_test/outputs_1_1_1/*depth_max.tif"

print_summary
```

### 5.3 New: `phase5.sh` — viz subcommand
```bash
#!/usr/bin/env bash
source "$(dirname "$0")/helpers.sh"

# Phase 5: Visualization (requires simulation output)
test_step 1 "Install viz dependencies" \
    "pip install run_anuga[viz]"

test_step 2 "Generate depth video" \
    "run-anuga viz /tmp/workdir/small_test/ --result-type depth"

test_step 3 "Verify video output" \
    "test -f /tmp/workdir/small_test/outputs_1_1_1/*depth*.mp4"

print_summary
```

### 5.4 PyInstaller smoke test improvements
Add to `release.yml`:
```yaml
- name: Smoke test without Python on PATH
  env:
    PATH: /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  run: |
    ./dist/run-anuga --help
    ./dist/run-anuga validate examples/small_test/
```

---

## Shared Test Infrastructure

### `conftest.py` — shared fixtures and auto-skip
```python
import os
import json
import shutil
import pytest

# ── Dependency detection ──────────────────────────────────────────
def pytest_configure(config):
    config.addinivalue_line("markers", "requires_anuga: needs ANUGA installed")
    config.addinivalue_line("markers", "requires_geo: needs shapely/rasterio")
    config.addinivalue_line("markers", "slow: takes > 30s")
    config.addinivalue_line("markers", "mpi: needs mpirun with multiple processes")

    try:
        import anuga
        config._anuga_available = True
    except ImportError:
        config._anuga_available = False

    try:
        import shapely, rasterio
        config._geo_available = True
    except ImportError:
        config._geo_available = False


def pytest_collection_modifyitems(config, items):
    if not getattr(config, "_anuga_available", False):
        skip = pytest.mark.skip(reason="ANUGA not installed")
        for item in items:
            if "requires_anuga" in item.keywords:
                item.add_marker(skip)

    if not getattr(config, "_geo_available", False):
        skip = pytest.mark.skip(reason="geo deps (shapely/rasterio) not installed")
        for item in items:
            if "requires_geo" in item.keywords:
                item.add_marker(skip)


# ── Fixtures ──────────────────────────────────────────────────────
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "data", "minimal_package")
SMALL_TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "small_test")


@pytest.fixture
def scenario_package(tmp_path):
    """Create a minimal scenario package for unit tests."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (tmp_path / "scenario.json").write_text(json.dumps({
        "format_version": "1.0",
        "epsg": "EPSG:28355",
        "boundary": "boundary.geojson",
        "duration": 60,
        "id": 1,
        "project": 1,
        "run_id": 1,
    }))
    (inputs / "boundary.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[321000, 5812000], [321200, 5812000]]
            },
            "properties": {"name": "south", "boundary_type": "external"}
        }]
    }))
    return tmp_path


@pytest.fixture
def scenario_package_full(scenario_package):
    """Scenario package with optional inputs (friction, inflow, structure)."""
    inputs = scenario_package / "inputs"

    # Add friction
    (inputs / "friction.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [
                [[321000,5812000],[321100,5812000],[321100,5812100],[321000,5812100],[321000,5812000]]
            ]},
            "properties": {"mannings_n": 0.1}
        }]
    }))

    # Update scenario.json to reference friction
    cfg = json.loads((scenario_package / "scenario.json").read_text())
    cfg["friction"] = "friction.geojson"
    (scenario_package / "scenario.json").write_text(json.dumps(cfg))

    return scenario_package


@pytest.fixture
def small_test_copy(tmp_path):
    """Copy of examples/small_test for integration tests (avoids mutating repo)."""
    dst = tmp_path / "small_test"
    shutil.copytree(SMALL_TEST_DIR, str(dst))
    return dst


@pytest.fixture
def small_geotiff(tmp_path):
    """Create a tiny 10x10 GeoTIFF for testing raster operations."""
    rasterio = pytest.importorskip("rasterio")
    import numpy as np
    from rasterio.transform import from_bounds

    path = tmp_path / "dem.tif"
    data = np.full((10, 10), 50.0, dtype=np.float32)  # flat 50m elevation
    transform = from_bounds(321000, 5812000, 321100, 5812100, 10, 10)

    with rasterio.open(
        str(path), "w", driver="GTiff",
        height=10, width=10, count=1, dtype="float32",
        crs="EPSG:28355", transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path
```

---

## CI Configuration Changes

### `pyproject.toml` updates
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: tests that take > 30s",
    "requires_anuga: needs ANUGA installed",
    "requires_geo: needs shapely/rasterio/geopandas",
    "mpi: needs mpirun with multiple processes",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.1",
    "jsonschema>=4.17",
    "hypothesis>=6.0",
    "syrupy>=4.0",
]
```

### `.github/workflows/ci.yml` updates
```yaml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.10", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v -m "not requires_anuga and not requires_geo and not mpi"

  test-geo:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev,sim]"
      - run: pytest tests/ -v -m "requires_geo and not requires_anuga"
```

---

## Code Changes Required

### 1. Extract `compute_yieldstep()` from `run.py`

Move the yieldstep calculation (lines 244-251 of `run.py`) into a standalone function:

```python
# In run_utils.py or a new module
def compute_yieldstep(duration):
    """Calculate yield step interval for the simulation evolve loop.

    Returns an integer number of seconds, clamped to
    [MIN_YIELDSTEP_S, MAX_YIELDSTEP_S].
    """
    base = math.floor(duration / defaults.MAX_YIELDSTEPS)
    yieldstep = max(base, defaults.MIN_YIELDSTEP_S)
    return min(yieldstep, defaults.MAX_YIELDSTEP_S)
```

### 2. Make pure-logic functions importable without ANUGA

Several functions in `run_utils.py` (`make_new_inflow`, `lookup_boundary_tag`, `is_dir_check`, `correction_for_polar_quadrants`) are already importable without ANUGA since `run_utils.py` only imports ANUGA lazily. Verify this works and add tests.

### 3. Add `conftest.py`

Create `tests/conftest.py` with the shared fixtures and auto-skip logic shown above.

---

## Test Execution Cheat Sheet

```bash
# Unit tests only (CI default, no heavy deps)
pytest tests/ -v -m "not requires_anuga and not requires_geo and not mpi"

# Unit + component tests (with geo deps installed)
pytest tests/ -v -m "not requires_anuga and not mpi"

# All tests including integration (ANUGA installed)
pytest tests/ -v -m "not mpi"

# MPI tests only (must use mpirun)
mpirun -np 2 python -m pytest tests/test_mpi.py -v

# Docker E2E tests
bash test-docker/test_readme.sh

# Quick smoke test for development
pytest tests/ -v -x --tb=short
```

---

## Implementation Priority

| Priority | Task | Tests Added | Effort |
|----------|------|-------------|--------|
| **P0** | Add `conftest.py` with fixtures + auto-skip | 0 (infrastructure) | Small |
| **P0** | Add pytest markers to `pyproject.toml` | 0 (infrastructure) | Tiny |
| **P1** | `test_package_loading.py` | 7+ | Small |
| **P1** | `test_data_transforms.py` | 8+ | Small |
| **P1** | `test_cli_resolve.py` | 5+ | Small |
| **P1** | Extract + test `compute_yieldstep()` | 5+ | Small |
| **P2** | `test_boundaries.py` (requires_geo) | 6+ | Medium |
| **P2** | `test_coordinate_checks.py` (requires_geo) | 6+ | Small |
| **P2** | `test_mesh_regions.py` (requires_geo) | 6+ | Small |
| **P2** | `test_raster_operations.py` (requires_geo) | 4+ | Medium |
| **P2** | `test_callbacks_http.py` | 4+ | Small |
| **P2** | Add Windows to CI matrix | 0 | Small |
| **P3** | `test_sim_lifecycle.py` (requires_anuga) | 5+ | Medium |
| **P3** | `test_post_process.py` (requires_anuga) | 4+ | Medium |
| **P3** | `test_checkpoint.py` (requires_anuga) | 2+ | Medium |
| **P3** | Snapshot tests for CLI output | 3+ | Small |
| **P4** | `test_mpi.py` (requires_anuga + mpirun) | 3+ | Large |
| **P4** | Docker `phase4.sh` (MPI E2E) | 2 | Medium |
| **P4** | Docker `phase5.sh` (viz E2E) | 3 | Medium |
| **P4** | Property-based tests (hypothesis) | 5+ | Medium |

**Estimated total: ~80 new tests, bringing the suite from 70 to ~150.**

---

## File Structure After Implementation

```
tests/
├── conftest.py                  # NEW: shared fixtures, auto-skip
├── data/
│   └── minimal_package/         # existing fixture
│       ├── scenario.json
│       └── inputs/boundary.geojson
├── test_callbacks.py            # existing (15 tests)
├── test_callbacks_http.py       # NEW: HydrataCallback HTTP edge cases
├── test_cli.py                  # existing (8 tests)
├── test_cli_resolve.py          # NEW: resolve_package_dir edge cases
├── test_config.py               # existing (14 tests)
├── test_coordinate_checks.py    # NEW: geometry validation (requires_geo)
├── test_data_transforms.py      # NEW: pure-logic run_utils functions
├── test_defaults.py             # existing (10 tests)
├── test_imports.py              # existing (8 tests)
├── test_logging_setup.py        # existing (17 tests)
├── test_boundaries.py           # NEW: boundary polygon creation (requires_geo)
├── test_mesh_regions.py         # NEW: mesh region extraction (requires_geo)
├── test_mpi.py                  # NEW: MPI parallel tests (requires_anuga + mpi)
├── test_package_loading.py      # NEW: _load_package_data tests
├── test_polar_quadrants.py      # NEW: angle correction function
├── test_post_process.py         # NEW: GeoTIFF from SWW (requires_anuga)
├── test_raster_operations.py    # NEW: raster clip/burn (requires_geo)
├── test_sim_lifecycle.py        # NEW: simulation phases (requires_anuga)
├── test_checkpoint.py           # NEW: checkpoint/restart (requires_anuga)
├── test_integration.py          # existing (1 test, can deprecate)
└── test_yieldstep.py            # NEW: yieldstep calculation

test-docker/
├── helpers.sh                   # existing
├── test_readme.sh               # existing orchestrator
├── phase1.sh                    # existing: core install
├── phase2.sh                    # existing: sim run
├── phase3.sh                    # existing: post-process
├── phase4.sh                    # NEW: MPI parallel run
└── phase5.sh                    # NEW: viz subcommand
```
