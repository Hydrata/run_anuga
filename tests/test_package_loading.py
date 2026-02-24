"""Tests for _load_package_data() from run_utils.py."""

import json
import os

import pytest

from run_anuga.run_utils import _load_package_data


class TestLoadPackageData:
    def test_loads_minimal_package(self, scenario_package):
        data = _load_package_data(str(scenario_package))
        assert "scenario_config" in data
        assert "run_label" in data
        assert "output_directory" in data
        assert "boundary" in data

    def test_run_label_format(self, scenario_package):
        data = _load_package_data(str(scenario_package))
        assert data["run_label"] == "run_1_1_1"

    def test_creates_output_directory(self, scenario_package):
        data = _load_package_data(str(scenario_package))
        assert os.path.isdir(data["output_directory"])

    def test_creates_checkpoint_directory(self, scenario_package):
        data = _load_package_data(str(scenario_package))
        assert os.path.isdir(data["checkpoint_directory"])
        assert data["checkpoint_directory"].rstrip("/\\").endswith("checkpoints")

    def test_output_directory_path(self, scenario_package):
        data = _load_package_data(str(scenario_package))
        expected = os.path.join(str(scenario_package), "outputs_1_1_1")
        assert data["output_directory"] == expected

    def test_mesh_filepath_extension(self, scenario_package):
        data = _load_package_data(str(scenario_package))
        # Mesh file is named after the run label: run_<project>_<id>_<run_id>.msh
        assert data["mesh_filepath"].endswith("run_1_1_1.msh")

    def test_missing_scenario_json(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="scenario.json"):
            _load_package_data(str(tmp_path))

    def test_loads_boundary_geojson(self, scenario_package):
        data = _load_package_data(str(scenario_package))
        assert data["boundary"]["type"] == "FeatureCollection"
        assert len(data["boundary"]["features"]) == 4

    def test_boundary_filename_set(self, scenario_package):
        data = _load_package_data(str(scenario_package))
        # Boundary file is always under the inputs/ subdirectory
        assert data["boundary_filename"].endswith(os.path.join("inputs", "boundary.geojson"))

    def test_optional_inputs_absent(self, scenario_package):
        data = _load_package_data(str(scenario_package))
        assert "friction" not in data
        assert "structure" not in data

    def test_optional_inputs_present(self, scenario_package_full):
        data = _load_package_data(str(scenario_package_full))
        assert "friction" in data
        assert "inflow" in data
        assert "structure" in data

    def test_resolution_from_config(self, scenario_package):
        # Scenario has resolution: 10 via minimal_package fixture doesn't set it,
        # but our fixture sets it in from_package... let's check the raw config.
        # Actually our fixture doesn't set resolution. Let's add it.
        cfg = json.loads((scenario_package / "scenario.json").read_text())
        cfg["resolution"] = 5.0
        (scenario_package / "scenario.json").write_text(json.dumps(cfg))

        data = _load_package_data(str(scenario_package))
        assert data["resolution"] == 5.0

    def test_no_resolution_in_config(self, scenario_package):
        data = _load_package_data(str(scenario_package))
        assert "resolution" not in data

    def test_idempotent_directory_creation(self, scenario_package):
        """Calling twice doesn't error."""
        _load_package_data(str(scenario_package))
        _load_package_data(str(scenario_package))

    def test_run_label_with_different_ids(self, tmp_path):
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        (tmp_path / "scenario.json").write_text(json.dumps({
            "epsg": "EPSG:28355",
            "boundary": "boundary.geojson",
            "duration": 60,
            "id": 42,
            "project": 7,
            "run_id": 3,
        }))
        (inputs / "boundary.geojson").write_text(json.dumps({
            "type": "FeatureCollection",
            "features": []
        }))
        data = _load_package_data(str(tmp_path))
        assert data["run_label"] == "run_7_42_3"
