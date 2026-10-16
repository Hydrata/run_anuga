"""True per-step peaks (TASK-3772, epic 3585).

Every "max" product used to be the max over the SAVED frames (one frame per
yieldstep), so a peak that rose and fell between two yields never showed up.
The anuga_core fork's ``Collect_max_quantities_operator`` tracks the running
maxima of stage, depth, speed and momentum magnitude at every timestep (on the
GPU without a per-step device->host copy) and, with ``store_to_sww=True``,
writes them into the SWW as the centroid arrays ``max_stage_c``,
``max_depth_c``, ``max_speed_c`` and ``max_uh_c`` (decision D38).
``sww_merge`` carries those arrays through to the merged SWW like any other
static centroid quantity.

This module holds the run-side wiring (:func:`attach_peak_operator`) and the
read-side helpers the rasters and the playback store share
(:func:`read_engine_peaks`, :func:`centroid_to_node`).

Products that stay frame-based (time of peak, run_diagnostics max_depth_m /
max_speed_ms, run_summary flow.*, a re-export of an SWW written before this
change) carry :data:`FRAME_MAX_LABEL`.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from run_anuga import defaults

logger = logging.getLogger(__name__)

#: A/B switch. Default ON; ``RUN_ANUGA_TRUE_PEAKS=0`` leaves the operator off
#: (the run is then exactly the pre-TASK-3772 frame-max behaviour).
TRUE_PEAKS_ENV = "RUN_ANUGA_TRUE_PEAKS"
_OFF_VALUES = {"0", "off", "false", "no"}

#: Update the maxima every N timesteps. GPU: every step (the kernel is cheap and
#: stays on the device). CPU: every 10th step (four numpy passes per update).
GPU_UPDATE_FREQUENCY = 1
CPU_UPDATE_FREQUENCY = 10
_MULTIPROCESSOR_GPU = 2

#: The engine's centroid arrays in the SWW (fork sww.py flag=4 quantities).
ENGINE_PEAK_VARIABLES = ("max_stage_c", "max_depth_c", "max_speed_c", "max_uh_c")

#: post_process_sww raster quantity -> engine array.
RASTER_QUANTITY_TO_ENGINE = {
    "depth": "max_depth_c",
    "velocity": "max_speed_c",
    "stage": "max_stage_c",
    "depthIntegratedVelocity": "max_uh_c",
}
#: Raster quantities whose engine raster is combined cell-wise with the
#: frame-max raster (never below a saved frame). depthIntegratedVelocity is
#: NOT: its frame-max raster is ||(max uh, max vh)||, components peaked at
#: different times, so it can exceed any real value; the engine max_uh is the
#: true per-step max of ||(uh, vh)|| and replaces it.
RASTER_COMBINE_WITH_FRAMES = frozenset({"depth", "velocity", "stage"})

#: playback-store envelope quantity -> engine array.
ENVELOPE_TO_ENGINE = {
    "depth": "max_depth_c",
    "velocity": "max_speed_c",
    "div": "max_uh_c",
}

ENVELOPE_SOURCE_ENGINE = "engine_per_step"
ENVELOPE_SOURCE_FRAMES = "saved_frames"
ENVELOPE_SOURCES = (ENVELOPE_SOURCE_ENGINE, ENVELOPE_SOURCE_FRAMES)

#: Label carried by every product that is still a max over saved frames.
FRAME_MAX_LABEL = "max over saved frames"


def true_peaks_enabled() -> bool:
    """False only when ``RUN_ANUGA_TRUE_PEAKS`` is set to an off value."""
    raw = os.environ.get(TRUE_PEAKS_ENV, "").strip().lower()
    return raw not in _OFF_VALUES


def attach_peak_operator(domain, multiprocessor_mode, *, log=None):
    """Attach (or re-use) the engine per-step max operator on ``domain``.

    Call on EVERY rank, after ``set_boundary`` and before ``evolve``, for a
    fresh or a resumed domain: ``set_collect_max_quantities`` returns the
    operator already on the domain (e.g. restored from a checkpoint pickle,
    which carries its host maxima) instead of creating a second one.

    Returns the operator, or None when switched off by
    :data:`TRUE_PEAKS_ENV` or the engine has no such operator (an older
    anuga) — the run then simply keeps the frame-max products.
    """
    log = log or logger
    if not true_peaks_enabled():
        log.info("true peaks: OFF (%s=%r)", TRUE_PEAKS_ENV, os.environ.get(TRUE_PEAKS_ENV))
        return None
    setter = getattr(domain, "set_collect_max_quantities", None)
    if setter is None:
        log.warning("true peaks: engine has no set_collect_max_quantities; frame-max products only")
        return None
    if not hasattr(domain, "max_quantities_operator"):
        # A checkpoint pickled by an engine that predates the attribute.
        domain.max_quantities_operator = None
    update_frequency = GPU_UPDATE_FREQUENCY if multiprocessor_mode == _MULTIPROCESSOR_GPU else CPU_UPDATE_FREQUENCY
    op = setter(
        update_frequency=update_frequency,
        velocity_zero_height=defaults.MINIMUM_STORABLE_HEIGHT_M,
        store_to_sww=True,
    )
    log.info(
        "true peaks: ON (update_frequency=%s, velocity_zero_height=%s m)",
        getattr(op, "update_frequency", update_frequency),
        getattr(op, "velocity_zero_height", defaults.MINIMUM_STORABLE_HEIGHT_M),
    )
    return op


def read_engine_peaks(sww_path):
    """The engine's per-step max centroid arrays from an SWW, or None.

    None when the SWW predates TASK-3772 (or the run had true peaks off): any
    of :data:`ENGINE_PEAK_VARIABLES` missing, or not 1-D of length
    ``number_of_volumes``. Values come back as float64, masked cells as NaN.
    """
    import netCDF4

    with netCDF4.Dataset(str(sww_path)) as ds:
        return engine_peaks_from_dataset(ds)


def engine_peaks_from_dataset(ds):
    """:func:`read_engine_peaks` on an already-open netCDF4 Dataset."""
    if not all(name in ds.variables for name in ENGINE_PEAK_VARIABLES):
        return None
    n_tri = ds.variables["volumes"].shape[0]
    out = {}
    for name in ENGINE_PEAK_VARIABLES:
        var = ds.variables[name]
        if var.ndim != 1 or var.shape[0] != n_tri:
            logger.warning("true peaks: %s has shape %s, expected (%d,); ignoring", name, var.shape, n_tri)
            return None
        out[name] = np.ma.filled(np.ma.asarray(var[:], dtype=np.float64), np.nan)
    return out


def centroid_to_node(values, volumes, n_node):
    """Per-triangle values -> per-node values by averaging the triangles that
    share each node (the same vertex averaging a smoothed SWW uses). Nodes no
    triangle references get 0.0; NaN triangles are left out of the average.
    """
    values = np.asarray(values, dtype=np.float64)
    volumes = np.asarray(volumes)
    tri_vals = np.repeat(values, 3)
    flat = volumes.reshape(-1)
    ok = np.isfinite(tri_vals)
    total = np.bincount(flat[ok], weights=tri_vals[ok], minlength=n_node)
    count = np.bincount(flat[ok], minlength=n_node)
    out = np.zeros(n_node, dtype=np.float64)
    np.divide(total, count, out=out, where=count > 0)
    return out


def read_centroid_xy(sww_path):
    """Georeferenced triangle centroids (x, y) of an SWW: the mean of each
    triangle's three vertices plus xllcorner/yllcorner — the same points
    ``plot_utils.get_centroids`` grids the frame-max rasters from.
    """
    import netCDF4

    with netCDF4.Dataset(str(sww_path)) as ds:
        x = np.asarray(ds.variables["x"][:], dtype=np.float64)
        y = np.asarray(ds.variables["y"][:], dtype=np.float64)
        volumes = np.asarray(ds.variables["volumes"][:], dtype=np.int64)
        xll = float(getattr(ds, "xllcorner", 0.0))
        yll = float(getattr(ds, "yllcorner", 0.0))
    cx = x[volumes].mean(axis=1) + xll
    cy = y[volumes].mean(axis=1) + yll
    return cx, cy
