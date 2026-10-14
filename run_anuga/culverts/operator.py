"""The culvert coupler (TASK-3642, epic 3585 W3): run_anuga hands the culverts to
``anuga_drainage.culverts`` (design 1: live SWMM 5.2 barrel + a fractional-step
operator moving water between the two D15 exchange regions).

``attach_culvert_coupler`` is called once, after distribute() and the inflow
operators. ``anuga`` and ``anuga_drainage`` load lazily, so importing this module
needs neither.
"""

from __future__ import annotations

import logging
import os

from run_anuga._imports import import_optional
from run_anuga.culverts import constants
from run_anuga.culverts.checks import ENDS
from run_anuga.culverts.features import CulvertBuildError

logger = logging.getLogger(__name__)


def swmm_workdir(output_directory) -> str:
    return os.path.join(output_directory, constants.METRICS_DIRNAME, constants.SWMM_WORKDIR_NAME)


def region_bed_m(anuga, domain, polygon):
    """Area-weighted mean bed elevation of the triangles whose centroid is in ``polygon``, or None."""
    numpy = import_optional("numpy")
    indices = anuga.Region(domain, polygon=polygon).indices
    if len(indices) == 0:
        return None
    elevation = domain.quantities["elevation"].centroid_values[indices]
    areas = domain.areas[indices]
    return float(numpy.sum(elevation * areas) / numpy.sum(areas))


def _open_ring(ring):
    ring = [list(point) for point in ring]
    return ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring


def headwall_culvert_dict(anuga, domain, result) -> dict:
    """The anuga_drainage D10 dict for one analysed culvert.

    Inverts the package omits default to the area-weighted bed of the end's exchange
    region (the Boyd convention). The exchange polygons are run_anuga's D15 regions.
    """
    culvert = result.culvert
    polygons = {end: _open_ring(result.regions[end].polygon) for end in ENDS}
    beds = {end: region_bed_m(anuga, domain, polygons[end]) for end in ENDS}
    for end in ENDS:
        if beds[end] is None:
            raise CulvertBuildError(
                f"{culvert.label}: the {end} exchange region holds no triangle on the run's mesh."
            )
    spec = {
        "name": culvert.name,
        "shape": culvert.shape,
        "coordinates": [list(point) for point in culvert.coordinates],
        "upstream_invert_m": culvert.upstream_invert_m if culvert.upstream_invert_m is not None else beds["inlet"],
        "downstream_invert_m": (
            culvert.downstream_invert_m if culvert.downstream_invert_m is not None else beds["outlet"]
        ),
        "barrels": culvert.barrels,
        "wall_thickness_m": culvert.wall_thickness_m,
        "blockage_pct": culvert.blockage_pct,
        "inlet_region": polygons["inlet"],
        "outlet_region": polygons["outlet"],
    }
    for key in ("width_m", "height_m", "diameter_m", "entrance_type", "material", "manning_n",
                "length_override_m"):
        value = getattr(culvert, key)
        if value is not None:
            spec[key] = value
    return spec


def attach_culvert_coupler(domain, results, output_directory, duration_s, log=None):
    """Build the SWMM coupler for ``results`` (CulvertMeshResult list) and attach it to ``domain``.

    Returns the coupler (use ``.finalize()`` / ``.timeseries(i)`` after the evolve loop).
    Raises CulvertBuildError (plain words) when a culvert cannot be coupled.
    """
    log = log or logger
    anuga = import_optional("anuga")
    drainage = import_optional("anuga_drainage.culverts", extra="sim")
    if getattr(anuga, "numprocs", 1) > 1 or getattr(domain, "numproc", 1) > 1:
        raise CulvertBuildError(
            "Culverts are not supported in multi-process (MPI) runs yet; run this scenario on a single process."
        )
    specs = [headwall_culvert_dict(anuga, domain, result) for result in results]
    try:
        culverts = [drainage.HeadwallCulvert.from_dict(spec) for spec in specs]
    except ValueError as error:
        raise CulvertBuildError(f"Cannot build the culverts: {error}") from error
    coupler = drainage.attach_culverts(
        domain,
        culverts,
        engine="swmm",
        workdir=swmm_workdir(output_directory),
        duration_s=float(duration_s) + constants.COUPLER_DURATION_MARGIN_S,
        smoothing_timescale=constants.COUPLER_SMOOTHING_TIMESCALE_S,
        use_momentum_jet=constants.COUPLER_USE_MOMENTUM_JET,
    )
    for culvert in culverts:
        log.info(
            "culvert coupler: %s, %s x%d, length %.2f m, inverts %.3f -> %.3f m (SWMM 5.2 barrel, headwall coupling)",
            culvert.name, culvert.shape, culvert.barrels, culvert.length_m,
            culvert.upstream_invert_m, culvert.downstream_invert_m,
        )
    return coupler
