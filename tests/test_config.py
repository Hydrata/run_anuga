"""Tests for run_anuga.config — Pydantic ScenarioConfig model."""

import json
import os

import pytest

from run_anuga.config import ScenarioConfig


def _minimal(**overrides):
    """Return a minimal valid config dict."""
    base = {
        "epsg": "EPSG:28355",
        "boundary": "boundary.geojson",
        "duration": 3600,
    }
    base.update(overrides)
    return base


class TestScenarioConfigDefaults:
    def test_minimal_config(self):
        cfg = ScenarioConfig(**_minimal())
        assert cfg.epsg == "EPSG:28355"
        assert cfg.duration == 3600
        assert cfg.id == 0
        assert cfg.project == 0
        assert cfg.run_id == 0

    def test_defaults_for_standalone(self):
        cfg = ScenarioConfig(**_minimal())
        assert cfg.format_version == "1.0"
        assert cfg.simplify_mesh is False
        assert cfg.store_mesh is False
        assert cfg.name is None
        assert cfg.elevation is None

    def test_all_fields(self):
        cfg = ScenarioConfig(**_minimal(
            id=1, project=42, run_id=7,
            name="Test", description="A test",
            control_server="https://hydrata.com",
            elevation="dem.tif",
            friction="friction.geojson",
            inflow="inflow.geojson",
            structure="structure.geojson",
            mesh_region="mesh.geojson",
            hydrology_status="complete",
            catchment="catchment.geojson",
            nodes="nodes.geojson",
            links="links.geojson",
            simplify_mesh=True,
            store_mesh=True,
            resolution=5.0,
            max_rmse_tolerance=0.5,
            model_start="2024-01-01T00:00:00Z",
        ))
        assert cfg.id == 1
        assert cfg.project == 42
        assert cfg.simplify_mesh is True
        assert cfg.resolution == 5.0


class TestScenarioConfigValidation:
    def test_format_version_1_0_passes(self):
        cfg = ScenarioConfig(**_minimal(format_version="1.0"))
        assert cfg.format_version == "1.0"

    def test_format_version_2_0_rejected(self):
        with pytest.raises(Exception, match="format_version"):
            ScenarioConfig(**_minimal(format_version="2.0"))

    def test_missing_required_epsg(self):
        with pytest.raises(Exception):
            ScenarioConfig(boundary="b.geojson", duration=100)

    def test_missing_required_boundary(self):
        with pytest.raises(Exception):
            ScenarioConfig(epsg="EPSG:28355", duration=100)

    def test_missing_required_duration(self):
        with pytest.raises(Exception):
            ScenarioConfig(epsg="EPSG:28355", boundary="b.geojson")

    def test_extra_fields_allowed(self):
        """Unknown fields should be preserved (forward compat)."""
        cfg = ScenarioConfig(**_minimal(custom_field="hello"))
        assert cfg.model_extra["custom_field"] == "hello"


class TestRunLabel:
    def test_default_run_label(self):
        cfg = ScenarioConfig(**_minimal())
        assert cfg.run_label == "run_0_0_0"

    def test_run_label_with_ids(self):
        cfg = ScenarioConfig(**_minimal(id=5, project=42, run_id=3))
        assert cfg.run_label == "run_42_5_3"


class TestFromPackage:
    def test_from_package_loads(self, tmp_path):
        data = _minimal(id=1, project=42, run_id=7)
        scenario_path = tmp_path / "scenario.json"
        scenario_path.write_text(json.dumps(data))
        cfg = ScenarioConfig.from_package(str(tmp_path))
        assert cfg.project == 42
        assert cfg.id == 1

    def test_from_package_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="scenario.json"):
            ScenarioConfig.from_package(str(tmp_path))

    def test_from_package_invalid_json(self, tmp_path):
        (tmp_path / "scenario.json").write_text("not json")
        with pytest.raises(Exception):
            ScenarioConfig.from_package(str(tmp_path))

    def test_from_package_minimal_fixture(self):
        """Load the existing minimal test fixture."""
        fixture_dir = os.path.join(
            os.path.dirname(__file__), "data", "minimal_package"
        )
        cfg = ScenarioConfig.from_package(fixture_dir)
        assert cfg.epsg == "EPSG:28355"
        assert cfg.duration == 600
        assert cfg.run_label == "run_1_1_1"


class TestModelDump:
    def test_roundtrip(self):
        data = _minimal(id=3, project=10, run_id=2, name="Roundtrip")
        cfg = ScenarioConfig(**data)
        dumped = cfg.model_dump()
        cfg2 = ScenarioConfig(**dumped)
        assert cfg2.id == cfg.id
        assert cfg2.name == cfg.name
        assert cfg2.run_label == cfg.run_label
