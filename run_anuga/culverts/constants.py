"""Culvert constants (TASK-3638, epic 3585 D15/D16).

The two coarse-mesh warning thresholds are PROVISIONAL (1 and 3) until the
W2a-4 calibration series (TASK-3632, Zaragoza mesh-sensitivity benchmark)
measures where the headwater error crosses 30 mm and 100 mm; that task
rewrites them here.
"""

#: The scenario.json ``requires`` capability and the package input key.
CULVERT_CAPABILITY = "culvert"

#: Coarseness ratio = local cell edge / culvert footprint width. Above WARN the
#: build logs a warning; above SEVERE it says the culvert result is unreliable.
#: PROVISIONAL — calibrated by TASK-3632 (W2a-4).
COARSE_RATIO_WARN = 1.0
COARSE_RATIO_SEVERE = 3.0

#: D15 exchange region sizing, in multiples of the local cell edge.
REGION_MIN_WIDTH_CELL_EDGES = 2.0
REGION_DEPTH_CELL_EDGES = 2.0
#: An exchange region must hold at least this many triangles (else the build fails).
MIN_REGION_TRIANGLES = 4
#: Each expansion step scales the region width and depth by this factor ...
REGION_EXPANSION_FACTOR = 1.25
#: ... at most this many times, before the build gives up and fails.
MAX_REGION_EXPANSIONS = 8

#: A culvert shorter than this many local cell edges gets a build WARNING
#: (operator ruling 13-10-26, TASK-3669): its two exchange regions sample the
#: same few cells. It never fails the build.
MIN_LENGTH_CELL_EDGES = 3.0

#: D10 per-feature defaults filled in when the package omits them.
DEFAULT_WALL_THICKNESS_M = 0.20
DEFAULT_BARRELS = 1
MAX_BARRELS = 20
DEFAULT_BLOCKAGE_PCT = 0.0
BLOCKAGE_METHOD = "reduced-area"
SHAPES = ("box", "circular")

#: Implied-timestep estimate for the coarse-mesh warning: ANUGA's CFL step is
#: about inradius / (|u| + sqrt(g h)); for the right-isosceles triangles a
#: max-area constraint produces, inradius = edge * (2 - sqrt(2)) / 2. The
#: reference depth is the culvert rise (height or diameter), floored here.
GRAVITY_MS2 = 9.81
MIN_REFERENCE_DEPTH_M = 0.5

#: culverts/metrics.json (the PINNED hydrata <-> run_anuga contract).
METRICS_SCHEMA_VERSION = 1
METRICS_DIRNAME = "culverts"
METRICS_FILENAME = "metrics.json"
TIMESERIES_HEADER = ("time_s", "q_m3s", "hw_level_m", "tw_level_m")

#: The SWMM coupler (TASK-3642, design 1): the fractional-step operator eases the
#: exchanged discharge toward SWMM's value over this timescale, and each outflow is
#: injected with a momentum jet along the barrel direction (both as validated in the
#: Towradgi spike, TASK-3634).
COUPLER_SMOOTHING_TIMESCALE_S = 30.0
COUPLER_USE_MOMENTUM_JET = True
#: SWMM is opened for the run duration plus this margin (it must outlast the last ANUGA step).
COUPLER_DURATION_MARGIN_S = 60.0
#: Time series sampling: every second, thinned to at most this many rows per culvert.
TIMESERIES_MAX_ROWS = 10000
#: Subdirectory of <output>/culverts/ for the SWMM work files (.inp/.rpt/.out).
SWMM_WORKDIR_NAME = "swmm"
