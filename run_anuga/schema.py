"""
JSON Schema for scenario.json — the configuration file that drives an ANUGA simulation.

Fields correspond to ScenarioPackageSerializer in gn_anuga/serializers.py.
"""

SCENARIO_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Hydrata Scenario Package",
    "description": "Configuration for a single ANUGA flood simulation run.",
    "type": "object",
    "properties": {
        "format_version": {
            "type": "string",
            "description": "Schema version. Must be '1.0' if present.",
            "enum": ["1.0"],
        },
        "id": {
            "type": "integer",
            "description": "Scenario ID.",
        },
        "run_id": {
            "type": "integer",
            "description": "Run ID for this execution.",
        },
        "project": {
            "type": "integer",
            "description": "Project ID that owns this scenario.",
        },
        "epsg": {
            "type": "string",
            "description": "Coordinate reference system, e.g. 'EPSG:28355'.",
        },
        "name": {
            "type": "string",
            "description": "Human-readable scenario name.",
        },
        "description": {
            "type": ["string", "null"],
            "description": "Optional scenario description.",
        },
        "control_server": {
            "type": ["string", "null"],
            "description": "Base URL of the Hydrata control server.",
        },
        "elevation": {
            "type": ["string", "null"],
            "description": "Filename of the elevation raster (GeoTIFF) in inputs/.",
        },
        "boundary": {
            "type": ["string", "null"],
            "description": "Filename of the boundary GeoJSON in inputs/.",
        },
        "friction": {
            "type": ["string", "null"],
            "description": "Filename of the friction GeoJSON in inputs/.",
        },
        "inflow": {
            "type": ["string", "null"],
            "description": "Filename of the inflow GeoJSON in inputs/.",
        },
        "structure": {
            "type": ["string", "null"],
            "description": "Filename of the structure GeoJSON in inputs/.",
        },
        "mesh_region": {
            "type": ["string", "null"],
            "description": "Filename of the mesh region GeoJSON in inputs/.",
        },
        "hydrology_status": {
            "type": ["string", "null"],
            "description": "Status of hydrology pre-processing.",
        },
        "catchment": {
            "type": ["string", "null"],
            "description": "Filename of the catchment GeoJSON in inputs/.",
        },
        "nodes": {
            "type": ["string", "null"],
            "description": "Filename of the nodes GeoJSON in inputs/.",
        },
        "links": {
            "type": ["string", "null"],
            "description": "Filename of the links GeoJSON in inputs/.",
        },
        "simplify_mesh": {
            "type": ["boolean", "null"],
            "description": "If true, use the mesher binary for adaptive mesh simplification.",
        },
        "store_mesh": {
            "type": ["boolean", "null"],
            "description": "If true, export the generated mesh as a shapefile.",
        },
        "resolution": {
            "type": ["number", "null"],
            "description": "Base mesh resolution in metres.",
        },
        "max_rmse_tolerance": {
            "type": ["number", "null"],
            "description": "Maximum RMSE tolerance for mesher.",
        },
        "model_start": {
            "type": ["string", "null"],
            "description": "Simulation start time as ISO 8601 string.",
        },
        "duration": {
            "type": "integer",
            "description": "Simulation duration in seconds.",
        },
    },
    "required": ["id", "project", "epsg", "boundary", "duration"],
}


class ValidationError(Exception):
    """Raised when a scenario.json fails validation."""


def validate_scenario(scenario_config):
    """
    Validate a scenario configuration dict against the schema.

    Uses jsonschema if installed; otherwise falls back to basic field checks.
    Raises ValidationError on failure.
    """
    # Check format_version compatibility
    fv = scenario_config.get("format_version")
    if fv is not None and fv != "1.0":
        raise ValidationError(
            f"Unsupported format_version '{fv}'. This version of run_anuga supports '1.0'."
        )

    try:
        import jsonschema
        try:
            jsonschema.validate(instance=scenario_config, schema=SCENARIO_SCHEMA)
        except jsonschema.ValidationError as e:
            raise ValidationError(str(e)) from e
    except ImportError:
        # Fallback: check required fields only
        for field in SCENARIO_SCHEMA["required"]:
            if field not in scenario_config:
                raise ValidationError(f"Missing required field: '{field}'")
        # Basic type checks for key numeric fields
        if "duration" in scenario_config:
            if not isinstance(scenario_config["duration"], int):
                raise ValidationError("'duration' must be an integer (seconds)")
