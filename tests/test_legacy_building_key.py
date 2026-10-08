"""TASK-3586 (epic 3585 W1) — the building-footprint layer was called
``structure`` before the Structure -> Building rename.

A run package is a stored artefact (S3 / a run directory), so a package built
pre-rename still points at the building file under scenario.json's legacy
``structure`` key. ``_load_package_data`` must read it as ``building`` and the
three consumers (interior holes, Mannings patches, Raised heights) must produce
IDENTICAL output for both keys.
"""
import json
import logging

import pytest

from run_anuga.config import ScenarioConfig
from run_anuga.run_utils import (
    LEGACY_INPUT_KEYS,
    _load_package_data,
    make_frictions,
    make_raised_elevation_pairs,
)

BUILDINGS = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [
                [[321040, 5812040], [321060, 5812040], [321060, 5812060],
                 [321040, 5812060], [321040, 5812040]]
            ]},
            "properties": {"method": "Reflective", "name": "hole_1"},
        },
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [
                [[321070, 5812040], [321080, 5812040], [321080, 5812050],
                 [321070, 5812050], [321070, 5812040]]
            ]},
            "properties": {"method": "Mannings", "name": "rough_1"},
        },
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [
                [[321010, 5812010], [321020, 5812010], [321020, 5812020],
                 [321010, 5812020], [321010, 5812010]]
            ]},
            "properties": {"method": "Raised", "name": "raised_1", "raised_height": 3.5},
        },
    ],
}


def _package_with_building_key(scenario_package, key, filename):
    """Write the buildings file and point scenario.json at it under ``key``."""
    (scenario_package / "inputs" / filename).write_text(json.dumps(BUILDINGS))
    cfg = json.loads((scenario_package / "scenario.json").read_text())
    cfg[key] = filename
    (scenario_package / "scenario.json").write_text(json.dumps(cfg))
    return scenario_package


@pytest.fixture
def new_package(scenario_package):
    return _package_with_building_key(scenario_package, "building", "str_1_building_01.json")


@pytest.fixture
def legacy_package(tmp_path_factory):
    """A pre-rename package: same features, filed under the legacy 'structure' key."""
    import shutil
    base = tmp_path_factory.mktemp("legacy")
    (base / "inputs").mkdir()
    (base / "scenario.json").write_text(json.dumps({
        "format_version": "1.0",
        "epsg": "EPSG:28355",
        "boundary": "boundary.geojson",
        "duration": 600,
        "id": 1,
        "project": 1,
        "run_id": 1,
    }))
    (base / "inputs" / "boundary.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[321000, 5812000], [321100, 5812000]]},
            "properties": {"boundary": "Reflective"},
        }],
    }))
    shutil.rmtree(base / "outputs_1_1_1", ignore_errors=True)
    return _package_with_building_key(base, "structure", "str_1_structure_01.json")


class TestLegacyBuildingKey:
    def test_legacy_key_table_names_only_the_building_layer(self):
        assert LEGACY_INPUT_KEYS == {"building": "structure"}

    def test_scenario_config_accepts_both_keys(self):
        new = ScenarioConfig.model_validate({
            "epsg": "EPSG:28355", "boundary": "b.geojson", "duration": 1, "building": "x.json"})
        old = ScenarioConfig.model_validate({
            "epsg": "EPSG:28355", "boundary": "b.geojson", "duration": 1, "structure": "x.json"})
        assert new.building == "x.json" and new.structure is None
        assert old.structure == "x.json" and old.building is None

    def test_new_key_loads_under_building(self, new_package):
        data = _load_package_data(str(new_package))
        assert data["building"] == BUILDINGS
        assert data["building_filename"].endswith("str_1_building_01.json")
        assert "structure" not in data

    def test_legacy_key_loads_under_building_and_warns(self, legacy_package, caplog):
        with caplog.at_level(logging.WARNING, logger="run_anuga.run_utils"):
            data = _load_package_data(str(legacy_package))
        assert data["building"] == BUILDINGS
        assert data["building_filename"].endswith("str_1_structure_01.json")
        # The legacy key is NOT surfaced as its own input: one layer, one key.
        assert "structure" not in data
        assert any("legacy 'structure' key" in r.getMessage() for r in caplog.records)

    def test_new_key_wins_when_both_present(self, new_package):
        cfg = json.loads((new_package / "scenario.json").read_text())
        cfg["structure"] = "does_not_exist.json"
        (new_package / "scenario.json").write_text(json.dumps(cfg))
        data = _load_package_data(str(new_package))
        assert data["building_filename"].endswith("str_1_building_01.json")

    def test_mannings_patches_identical_for_both_keys(self, new_package, legacy_package):
        new = make_frictions(_load_package_data(str(new_package)))
        old = make_frictions(_load_package_data(str(legacy_package)))
        assert new == old
        # exactly one Mannings patch (+ the 'All' fallback) for the three features
        assert len(new) == 2 and new[-1][0] == "All"

    def test_raised_heights_identical_for_both_keys(self, new_package, legacy_package):
        new = make_raised_elevation_pairs(_load_package_data(str(new_package)))
        old = make_raised_elevation_pairs(_load_package_data(str(legacy_package)))
        assert new == old
        assert len(new) == 1 and new[0][1] == 3.5

    @pytest.mark.requires_geo
    def test_interior_holes_identical_for_both_keys(self, new_package, legacy_package):
        from run_anuga.run_utils import make_interior_holes_and_tags
        new_holes, new_tags = make_interior_holes_and_tags(_load_package_data(str(new_package)))
        old_holes, old_tags = make_interior_holes_and_tags(_load_package_data(str(legacy_package)))
        assert new_holes == old_holes and new_tags == old_tags
        assert len(new_holes) == 1  # the one Reflective footprint
