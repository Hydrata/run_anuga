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

#: TASK-3638 (epic 3585, D16): the package capabilities this runner implements.
#: A package lists what it needs in scenario.json ``requires``; a runner that
#: lacks one must refuse the package rather than silently drop the input (an
#: older runner ignores unknown keys and would report a culvert-free run as a
#: success).
SUPPORTED_CAPABILITIES = frozenset({"culvert"})


class UnsupportedCapabilityError(ValueError):
    """scenario.json ``requires`` names a capability this runner does not have."""


def check_required_capabilities(scenario_config) -> list:
    """Return the package's ``requires`` list; raise if any entry is unsupported.

    Accepts the raw scenario.json dict. A missing or empty ``requires`` passes.
    """
    requires = (scenario_config or {}).get("requires") or []
    if isinstance(requires, str) or not isinstance(requires, (list, tuple)):
        raise UnsupportedCapabilityError(
            f"scenario.json 'requires' must be a list of capability names, got {requires!r}"
        )
    missing = sorted({str(name) for name in requires} - SUPPORTED_CAPABILITIES)
    if missing:
        raise UnsupportedCapabilityError(
            f"This package requires capability {', '.join(repr(m) for m in missing)}, which this "
            f"version of run_anuga does not support (supported: {', '.join(sorted(SUPPORTED_CAPABILITIES))}). "
            "Refusing to run rather than silently ignore it: upgrade run_anuga."
        )
    return list(requires)


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
    rainfall: Optional[str] = None
    building: Optional[str] = None
    # TASK-3586 legacy package key (deprecated): packages written before the
    # Structure -> Building rename point at the building-footprint file under
    # 'structure'. _load_package_data reads it as 'building' when the new key
    # is absent, so a stored package on S3 re-runs unchanged.
    structure: Optional[str] = None
    mesh_region: Optional[str] = None
    hydrology_status: Optional[str] = None
    catchment: Optional[str] = None
    nodes: Optional[str] = None
    links: Optional[str] = None
    store_mesh: bool = False
    resolution: Optional[float] = None
    max_rmse_tolerance: Optional[float] = None
    model_start: Optional[str] = None
    # TASK-1271 W4.3 — breaklines for mesh edge conformance
    breakline: Optional[str] = None
    # TASK-3638 (epic 3585) — culvert layer file and the capability guard list.
    culvert: Optional[str] = None
    requires: list[str] = []
    # Default near-spacing (m) for breakline grading when no per-line value set.
    default_near_spacing: Optional[float] = 2.0

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

    @field_validator("requires")
    @classmethod
    def check_requires(cls, v: list[str]) -> list[str]:
        check_required_capabilities({"requires": v})
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
