"""
Pydantic configuration model for scenario.json.

Usage::

    from run_anuga.config import ScenarioConfig

    config = ScenarioConfig.from_package("/path/to/package")
    print(config.run_label)        # "run_42_1_7"
    print(config.model_dump())     # dict, suitable for JSON serialisation
"""

from __future__ import annotations

import json
import os
from typing import Optional

from pydantic import BaseModel, field_validator


class ScenarioConfig(BaseModel):
    """Validated configuration for an ANUGA flood simulation run."""

    format_version: str = "1.0"
    id: int = 0
    run_id: int = 0
    project: int = 0
    epsg: str
    boundary: str
    duration: int
    name: Optional[str] = None
    description: Optional[str] = None
    control_server: Optional[str] = None
    elevation: Optional[str] = None
    friction: Optional[str] = None
    inflow: Optional[str] = None
    structure: Optional[str] = None
    mesh_region: Optional[str] = None
    hydrology_status: Optional[str] = None
    catchment: Optional[str] = None
    nodes: Optional[str] = None
    links: Optional[str] = None
    simplify_mesh: bool = False
    store_mesh: bool = False
    resolution: Optional[float] = None
    max_rmse_tolerance: Optional[float] = None
    model_start: Optional[str] = None
    flow_algorithm: Optional[str] = None

    model_config = {"extra": "allow"}

    @field_validator("format_version")
    @classmethod
    def check_format_version(cls, v: str) -> str:
        if v != "1.0":
            raise ValueError(
                f"Unsupported format_version '{v}'. "
                "This version of run_anuga supports '1.0'."
            )
        return v

    @property
    def run_label(self) -> str:
        """Label used for output directories and filenames."""
        return f"run_{self.project}_{self.id}_{self.run_id}"

    @classmethod
    def from_package(cls, package_dir: str) -> "ScenarioConfig":
        """Load and validate scenario.json from a package directory."""
        scenario_path = os.path.join(package_dir, "scenario.json")
        if not os.path.isfile(scenario_path):
            raise FileNotFoundError(
                f'Could not find "scenario.json" in {package_dir}'
            )
        with open(scenario_path) as f:
            data = json.load(f)
        return cls.model_validate(data)
