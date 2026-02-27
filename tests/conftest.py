"""Shared fixtures and auto-skip logic for the run_anuga test suite."""

import json
import os
import shutil

import pytest

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "data", "minimal_package")
SMALL_TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "small_test")


# ── Dependency detection ──────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "requires_anuga: needs ANUGA installed")
    config.addinivalue_line("markers", "requires_geo: needs shapely/rasterio/geopandas")
    config.addinivalue_line("markers", "slow: takes > 30s")
    config.addinivalue_line("markers", "mpi: needs mpirun with multiple processes")

    try:
        import anuga  # noqa: F401
        config._anuga_available = True
    except ImportError:
        config._anuga_available = False

    try:
        import shapely  # noqa: F401
        import rasterio  # noqa: F401
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


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def scenario_package(tmp_path):
    """Create a minimal scenario package for unit tests."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (tmp_path / "scenario.json").write_text(json.dumps({
        "format_version": "1.0",
        "epsg": "EPSG:28355",
        "boundary": "boundary.geojson",
        "duration": 600,
        "id": 1,
        "project": 1,
        "run_id": 1,
    }))
    (inputs / "boundary.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "EPSG:28355"}
        },
        "features": [
            {
                "type": "Feature",
                "id": "boundary_1",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[321000.0, 5812000.0], [321100.0, 5812000.0]]
                },
                "properties": {"boundary": "Transmissive", "location": "External"}
            },
            {
                "type": "Feature",
                "id": "boundary_2",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[321100.0, 5812000.0], [321100.0, 5812100.0]]
                },
                "properties": {"boundary": "Reflective", "location": "External"}
            },
            {
                "type": "Feature",
                "id": "boundary_3",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[321100.0, 5812100.0], [321000.0, 5812100.0]]
                },
                "properties": {"boundary": "Transmissive", "location": "External"}
            },
            {
                "type": "Feature",
                "id": "boundary_4",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[321000.0, 5812100.0], [321000.0, 5812000.0]]
                },
                "properties": {"boundary": "Reflective", "location": "External"}
            },
        ]
    }))
    return tmp_path


@pytest.fixture
def scenario_package_full(scenario_package):
    """Scenario package with optional inputs (friction, inflow, structure, mesh_region)."""
    inputs = scenario_package / "inputs"

    # Add friction
    (inputs / "friction.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [
                [[321020, 5812020], [321080, 5812020], [321080, 5812080],
                 [321020, 5812080], [321020, 5812020]]
            ]},
            "properties": {"mannings": 0.1}
        }]
    }))

    # Add inflow
    (inputs / "inflow.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "id": "rainfall_1",
            "geometry": {"type": "Polygon", "coordinates": [
                [[321000, 5812000], [321100, 5812000], [321100, 5812100],
                 [321000, 5812100], [321000, 5812000]]
            ]},
            "properties": {"type": "Rainfall", "data": "10"}
        }]
    }))

    # Add structure
    (inputs / "structure.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [
                [[321040, 5812040], [321060, 5812040], [321060, 5812060],
                 [321040, 5812060], [321040, 5812040]]
            ]},
            "properties": {"method": "Mannings", "name": "building_1"}
        }]
    }))

    # Update scenario.json to reference optional inputs
    cfg = json.loads((scenario_package / "scenario.json").read_text())
    cfg["friction"] = "friction.geojson"
    cfg["inflow"] = "inflow.geojson"
    cfg["structure"] = "structure.geojson"
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

    path = tmp_path / "source_dem.tif"
    data = np.full((10, 10), 50.0, dtype=np.float32)
    transform = from_bounds(321000, 5812000, 321100, 5812100, 10, 10)

    with rasterio.open(
        str(path), "w", driver="GTiff",
        height=10, width=10, count=1, dtype="float32",
        crs="EPSG:28355", transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path
