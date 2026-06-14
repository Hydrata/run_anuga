"""TASK-1715 — Build-time Triangle-safety conditioner for breaklines.

Shewchuk Triangle needs a VALID PSLG: constraint segments may only intersect at
shared endpoints, must lie inside the bounding polygon, and must not contain
degenerate (near-zero-length) segments — otherwise Triangle errors out or emits
sub-CFL sliver triangles near the constraint.  ``condition_breaklines`` takes raw
user-drawn (or, later, auto-extracted) breakline geometry and returns
ANUGA-ready OPEN polylines in absolute coordinates::

    [ [[x, y], [x, y], ...], ... ]   # list of open polylines

The pipeline (TASK-1715 spec, decision D7 "conditioning"):

  1. clip each line to ``boundary_polygon`` (segments outside Triangle's domain
     are illegal / wasted),
  2. simplify then re-densify each line toward its ``near_spacing`` (a conformed
     edge should never be finer than the mesh wants),
  3. dedupe coincident vertices and drop sub-CFL / degenerate segments,
  4. NODE all lines at their crossings so PSLG segments meet only at shared
     endpoints (a hand-edit can re-introduce a crossing at any time).

It is pure and defensive: any single malformed line is skipped (logged), never
raised, so a bad hand-drawn line can never crash the mesh build.

The grading path (run_utils.make_breaklines buffer rings) is INDEPENDENT and is
NOT touched here — conform + grade compose.
"""

import logging
import math

logger = logging.getLogger(__name__)

# Default floor for a "sub-CFL / degenerate" segment, in CRS units (metres for
# the projected/UTM CRSs ANUGA meshes in).  Matches the sliver threshold used by
# the Reflective hole path (run_utils sliver-merge ~0.5m).  Overridable per call.
DEFAULT_MIN_SEGMENT_LENGTH = 0.5

# Simplify tolerance as a fraction of near_spacing — removes near-colinear jitter
# without moving the line off its true alignment.
SIMPLIFY_FRACTION = 0.25


def _iter_linestrings(geom):
    """Yield shapely LineStrings from any line-ish geometry (handles Multi*/empty)."""
    if geom is None or geom.is_empty:
        return
    gtype = geom.geom_type
    if gtype == 'LineString':
        yield geom
    elif gtype in ('MultiLineString', 'GeometryCollection'):
        for part in geom.geoms:
            yield from _iter_linestrings(part)
    # Points / Polygons from a degenerate intersection are silently ignored.


def _dedupe_and_drop_short(coords, min_segment_length):
    """Collapse vertices closer than min_segment_length to the previous KEPT vertex.

    Always keeps the first and last vertex so the line is never truncated; an
    interior vertex that would make a sub-floor segment is dropped (its segment is
    absorbed into the next), killing degenerate/sub-CFL micro-segments and exact
    duplicates in one pass.
    """
    if not coords:
        return []
    kept = [coords[0]]
    for pt in coords[1:-1]:
        if math.hypot(pt[0] - kept[-1][0], pt[1] - kept[-1][1]) >= min_segment_length:
            kept.append(pt)
    last = coords[-1]
    # Drop the final vertex onto the previous one if it would be a sub-floor seg,
    # but never drop below 2 vertices (a line needs an endpoint).
    if len(kept) >= 2 and \
            math.hypot(last[0] - kept[-1][0], last[1] - kept[-1][1]) < min_segment_length:
        kept[-1] = last  # snap the near-coincident tail onto the true endpoint
    else:
        kept.append(last)
    return kept


def _line_length(coords):
    return sum(math.hypot(coords[i + 1][0] - coords[i][0],
                          coords[i + 1][1] - coords[i][1])
               for i in range(len(coords) - 1))


def _simplify_densify(line, near_spacing):
    """Simplify near-colinear jitter, then densify so no segment exceeds near_spacing."""
    near_spacing = max(float(near_spacing), 1e-6)
    simplified = line.simplify(near_spacing * SIMPLIFY_FRACTION, preserve_topology=False)
    if simplified.is_empty or len(simplified.coords) < 2:
        return line
    return simplified.segmentize(near_spacing)


def condition_breaklines(breakline_geojson, boundary_polygon,
                         default_near_spacing=2.0, min_segment_length=None):
    """Condition raw breaklines into a valid Triangle PSLG.

    Parameters
    ----------
    breakline_geojson : dict or None
        A GeoJSON FeatureCollection of LineString breaklines (the same
        ``input_data['breakline']`` shape make_breaklines consumes).  Each feature
        may carry ``properties.near_spacing``.
    boundary_polygon : list of [x, y]
        The bounding polygon (absolute coords) the mesh is built in.  Lines are
        clipped to this.
    default_near_spacing : float
        Fallback target edge length for a line with no ``near_spacing`` property.
    min_segment_length : float or None
        Sub-CFL/degenerate floor.  Defaults to min(DEFAULT_MIN_SEGMENT_LENGTH,
        quarter of the finest near_spacing) so it never erases legitimate detail
        on a finely-spaced line.

    Returns
    -------
    list of polylines, each a list of [x, y] pairs (absolute coords), ready to
    pass as ``create_mesh_from_regions(breaklines=...)``.  Empty list if there is
    nothing valid to conform.
    """
    if not breakline_geojson or not breakline_geojson.get('features'):
        return []

    try:
        from shapely.geometry import shape as _shape, Polygon as _Polygon, \
            MultiLineString as _MultiLineString
        from shapely.ops import unary_union
    except ImportError:
        logger.warning("shapely not available — breakline conditioning skipped")
        return []

    try:
        boundary = _Polygon(boundary_polygon)
        if not boundary.is_valid:
            boundary = boundary.buffer(0)
    except Exception:
        logger.warning("condition_breaklines: invalid boundary_polygon — skipping all breaklines")
        return []

    # Resolve the sub-CFL floor from the finest near_spacing present.
    near_spacings = []
    for feature in breakline_geojson['features']:
        ns = (feature.get('properties') or {}).get('near_spacing') or default_near_spacing
        try:
            near_spacings.append(float(ns))
        except (TypeError, ValueError):
            near_spacings.append(default_near_spacing)
    finest = min(near_spacings) if near_spacings else default_near_spacing
    if min_segment_length is None:
        min_segment_length = max(1e-3, min(DEFAULT_MIN_SEGMENT_LENGTH, finest * 0.25))

    # 1+2+3: per-line clip -> simplify/densify -> dedupe/drop-short.
    conditioned = []
    for feature, near_spacing in zip(breakline_geojson['features'], near_spacings):
        geom = feature.get('geometry')
        if not geom:
            continue
        try:
            line = _shape(geom)
        except Exception:
            logger.warning(f"condition_breaklines: unparseable geometry for {feature.get('id')}")
            continue
        if line.is_empty or line.geom_type not in ('LineString', 'MultiLineString'):
            continue
        try:
            clipped = line.intersection(boundary)
        except Exception:
            logger.warning(f"condition_breaklines: clip failed for {feature.get('id')} — skipping")
            continue
        for part in _iter_linestrings(clipped):
            try:
                processed = _simplify_densify(part, near_spacing)
            except Exception:
                processed = part
            coords = _dedupe_and_drop_short([list(c[:2]) for c in processed.coords],
                                            min_segment_length)
            if len(coords) >= 2 and _line_length(coords) >= min_segment_length:
                conditioned.append(coords)

    if not conditioned:
        return []

    # 4: node all lines at their crossings so segments meet only at shared
    # endpoints.  unary_union planarizes the collection, inserting a vertex at
    # every intersection — the PSLG-validity guarantee Triangle requires.
    try:
        noded = unary_union(_MultiLineString(conditioned))
    except Exception:
        logger.warning("condition_breaklines: noding failed — using un-noded lines")
        noded = None

    result = []
    if noded is not None:
        for part in _iter_linestrings(noded):
            coords = _dedupe_and_drop_short([list(c[:2]) for c in part.coords],
                                            min_segment_length)
            if len(coords) >= 2 and _line_length(coords) >= min_segment_length:
                result.append(coords)
    else:
        result = conditioned

    return result
