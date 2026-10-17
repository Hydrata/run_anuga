"""The culvert coupler (TASK-3642, epic 3585 W3): run_anuga hands the culverts to
``anuga_drainage.culverts`` (design 1: live SWMM 5.2 barrel + a fractional-step
operator moving water between the two D15 exchange regions).

``attach_culvert_coupler`` is called once, after distribute() and the inflow
operators; it first lowers any exchange region whose lowest ground is above its invert
(D34, TASK-3700; sill rule TASK-3771).
``anuga`` and ``anuga_drainage`` load lazily, so importing this module needs neither.
"""

from __future__ import annotations

import logging
import os

from run_anuga._imports import import_optional
from run_anuga.culverts import constants
from run_anuga.culverts.checks import ENDS, buried_invert_warning
from run_anuga.culverts.features import CulvertBuildError
from run_anuga.culverts.metrics import write_culvert_metrics
from run_anuga.culverts.regions import carve_to_invert

logger = logging.getLogger(__name__)


def coupler_smoothing_timescale(log=None):
    """COUPLER_SMOOTHING_TIMESCALE_S, or the RUN_ANUGA_CULVERT_SMOOTHING_S override (TASK-3771)."""
    raw = os.environ.get(constants.COUPLER_SMOOTHING_ENV, "").strip()
    if not raw:
        return constants.COUPLER_SMOOTHING_TIMESCALE_S
    try:
        value = float(raw)
    except ValueError as error:
        raise CulvertBuildError(f"{constants.COUPLER_SMOOTHING_ENV} must be seconds, got {raw!r}") from error
    if not value >= 0.0:
        raise CulvertBuildError(f"{constants.COUPLER_SMOOTHING_ENV} must be >= 0, got {raw!r}")
    (log or logger).info("culvert coupler: smoothing timescale %.3g s (%s override)", value,
                         constants.COUPLER_SMOOTHING_ENV)
    return value


def coupler_equalisation_kwargs(log=None):
    """attach_culverts kwargs for the RUN_ANUGA_CULVERT_EQUALISATION override (TASK-3771);
    {} (the fork's default cap) when unset."""
    raw = os.environ.get(constants.COUPLER_EQUALISATION_ENV, "").strip().lower()
    if not raw or raw == "default":
        return {}
    if raw not in constants.COUPLER_EQUALISATION_MODES:
        raise CulvertBuildError(
            f"{constants.COUPLER_EQUALISATION_ENV} must be one of {constants.COUPLER_EQUALISATION_MODES}, got {raw!r}"
        )
    (log or logger).info("culvert coupler: equalisation cap %s (%s override)", raw, constants.COUPLER_EQUALISATION_ENV)
    if raw == "off":
        return {"equalisation_factor": None}
    return {"equalisation_strict": True}


def swmm_workdir(output_directory) -> str:
    return os.path.join(output_directory, constants.METRICS_DIRNAME, constants.SWMM_WORKDIR_NAME)


def region_bed_m(anuga, domain, polygon):
    """Lowest bed elevation of the triangles whose centroid is in ``polygon``, or None.

    TASK-3771: the default for an omitted invert. The coupler reads each end's stage as the
    level its water fills on the region's own ground (anuga_drainage LevelFill), so a dry
    region reads its lowest ground: with the invert there it reads zero head, and the culvert
    starts to flow as soon as water collects at the end. The area-weighted MEAN (the
    pre-TASK-3771 default) would hold the water back until it covered the region's average
    ground, which on a bed-plus-bank region is well above the channel.
    """
    indices = anuga.Region(domain, polygon=polygon).indices
    if len(indices) == 0:
        return None
    return float(min(domain.quantities["elevation"].centroid_values[indices]))


def _open_ring(ring):
    ring = [list(point) for point in ring]
    return ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring


def headwall_culvert_dict(anuga, domain, result) -> dict:
    """The anuga_drainage D10 dict for one analysed culvert.

    Inverts the package omits default to the LOWEST bed of the end's exchange region
    (TASK-3771; before it, the area-weighted mean). The exchange polygons are run_anuga's
    D15 regions.
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


def carve_exchange_regions(anuga, domain, results, log=None) -> list:
    """D34 (TASK-3700, amended TASK-3771): lower each end's exchange region by its sill so its
    lowest ground meets the user's invert (nothing is cut where the ground already reaches it).

    Ends whose invert the package omits are skipped (that invert IS the region's mean
    ground). Uses the SAME triangles the coupler exchanges through (anuga.Region on the
    D15 polygon). Logs every end's carve; an invert more than BURIED_INVERT_WARN_M below
    the region's mean ground adds a plain-English warning to the culvert's
    ``result.warnings`` (metrics.json). Returns [(result, CarveResult), ...].
    """
    log = log or logger
    carves = []
    for result in results:
        culvert = result.culvert
        for end in ENDS:
            invert = culvert.upstream_invert_m if end == "inlet" else culvert.downstream_invert_m
            if invert is None:
                continue
            indices = anuga.Region(domain, polygon=_open_ring(result.regions[end].polygon)).indices
            if len(indices) == 0:
                continue  # headwall_culvert_dict names this failure plainly
            carve = carve_to_invert(domain, indices, float(invert), end)
            carves.append((result, carve))
            log.info(
                "culvert carve: %s %s, invert %.3f m, region lowest ground %.3f m, mean ground %.3f -> %.3f m: "
                "%d of %d cell(s) lowered, max cut %.3f m, %.1f m3 of ground carved over %.1f m2",
                culvert.label, end, carve.invert_m, carve.lowest_before_m, carve.bed_before_m, carve.bed_after_m,
                carve.cells_lowered, len(indices), carve.max_cut_m, carve.carved_volume_m3, carve.carved_area_m2,
            )
            warning = buried_invert_warning(culvert, carve)
            if warning:
                result.warnings.append(warning)
                log.warning(warning)
    return carves


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
    # D34: carve BEFORE the coupler is built (it measures each region's mean ground once).
    carves = carve_exchange_regions(anuga, domain, results, log=log)
    if any(carve.buried_by_m > constants.BURIED_INVERT_WARN_M for _, carve in carves):
        write_culvert_metrics(output_directory, results)  # the build-time document gains the warning
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
        smoothing_timescale=coupler_smoothing_timescale(log),
        **coupler_equalisation_kwargs(log),
        use_momentum_jet=constants.COUPLER_USE_MOMENTUM_JET,
    )
    for culvert in culverts:
        log.info(
            "culvert coupler: %s, %s x%d, length %.2f m, inverts %.3f -> %.3f m (SWMM 5.2 barrel, headwall coupling)",
            culvert.name, culvert.shape, culvert.barrels, culvert.length_m,
            culvert.upstream_invert_m, culvert.downstream_invert_m,
        )
    return coupler
