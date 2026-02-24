"""
Default constants for ANUGA simulation parameters.

These values were previously hardcoded throughout run.py and run_utils.py.
Override them by passing explicit values in scenario.json or function arguments.
"""

# Structure / building parameters
BUILDING_BURN_HEIGHT_M = 5.0
"""Height (metres) added to the DEM where buildings are rasterised."""

BUILDING_MANNINGS_N = 10.0
"""Manning's roughness coefficient applied to building footprints."""

DEFAULT_MANNINGS_N = 0.04
"""Default Manning's roughness applied to all areas not covered by friction polygons."""

# Rainfall conversion
RAINFALL_FACTOR = 1.0 / (1000.0 * 3600.0)  # = 2.7778e-7
"""Conversion factor from mm/hr rainfall intensity to m/s for ANUGA rate operators.

Derivation: 1 mm/hr = 1/1000 m/hr = 1/1000 / 3600 m/s = 2.7778e-7 m/s.
Used as the `factor` argument to Polygonal_rate_operator when input data is in mm/hr.
"""

# Domain evolution
MINIMUM_STORABLE_HEIGHT_M = 0.005
"""Minimum water depth (metres) stored in SWW output files."""

MIN_ALLOWED_HEIGHT_M = 1.0e-05
"""Minimum water depth (metres) for velocity extrapolation in post-processing."""

MAX_YIELDSTEPS = 100
"""Maximum number of yield steps during domain evolution."""

MIN_YIELDSTEP_S = 60
"""Minimum yield step interval (seconds) — at most yield every minute."""

MAX_YIELDSTEP_S = 1800
"""Maximum yield step interval (seconds) — at least yield every 30 minutes."""

# Mesh generation
MAX_TRIANGLE_AREA = 10_000_000
"""Maximum triangle area (m^2) for the mesher config."""

# Post-processing
K_NEAREST_NEIGHBOURS = 3
"""Number of nearest neighbours for GeoTIFF interpolation."""

# External tool paths
DEFAULT_MESHER_EXE = "mesher"
"""Default name of the mesher binary.  Resolved via PATH or MESHER_EXE env var."""
