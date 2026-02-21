"""Tests for run_anuga.schema — JSON Schema validation of scenario.json."""

import pytest
from run_anuga.schema import SCENARIO_SCHEMA, validate_scenario, ValidationError


def _minimal_scenario(**overrides):
    """Return a minimal valid scenario config dict."""
    base = {
        "id": 1,
        "project": 42,
        "epsg": "EPSG:28355",
        "boundary": "boundary.geojson",
        "duration": 3600,
    }
    base.update(overrides)
    return base


class TestSchemaStructure:
    def test_schema_has_required_fields(self):
        assert "required" in SCENARIO_SCHEMA
        for field in ["id", "project", "epsg", "boundary", "duration"]:
            assert field in SCENARIO_SCHEMA["required"]

    def test_schema_has_format_version(self):
        assert "format_version" in SCENARIO_SCHEMA["properties"]

    def test_schema_has_all_serializer_fields(self):
        expected = [
            "id", "run_id", "project", "epsg", "name", "description",
            "control_server", "elevation", "boundary", "friction", "inflow",
            "structure", "mesh_region", "hydrology_status", "catchment",
            "nodes", "links", "simplify_mesh", "resolution",
            "max_rmse_tolerance", "model_start", "duration", "format_version",
        ]
        for field in expected:
            assert field in SCENARIO_SCHEMA["properties"], f"Missing field: {field}"


class TestValidateScenario:
    def test_minimal_valid_scenario(self):
        validate_scenario(_minimal_scenario())

    def test_format_version_1_0_passes(self):
        validate_scenario(_minimal_scenario(format_version="1.0"))

    def test_format_version_absent_passes(self):
        scenario = _minimal_scenario()
        assert "format_version" not in scenario
        validate_scenario(scenario)

    def test_format_version_2_0_rejected(self):
        with pytest.raises(ValidationError, match="format_version"):
            validate_scenario(_minimal_scenario(format_version="2.0"))

    def test_format_version_wrong_type_rejected(self):
        with pytest.raises(ValidationError, match="format_version"):
            validate_scenario(_minimal_scenario(format_version="0.9"))

    def test_missing_required_field_rejected(self):
        scenario = _minimal_scenario()
        del scenario["duration"]
        with pytest.raises(ValidationError):
            validate_scenario(scenario)

    def test_full_scenario_passes(self):
        scenario = _minimal_scenario(
            format_version="1.0",
            run_id=7,
            name="Test Scenario",
            description="A test",
            control_server="https://hydrata.com/",
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
            resolution=5.0,
            max_rmse_tolerance=0.5,
            model_start="2024-01-01T00:00:00Z",
        )
        validate_scenario(scenario)
