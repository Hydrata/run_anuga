"""Parse the package's culvert.geojson into validated Culvert records (TASK-3638).

Pure Python (no numpy/shapely): this runs inside ``_load_package_data``, which
must stay importable on a runner without the geo stack.

Each feature is a LineString drawn inlet -> outlet (first vertex = Inlet,
last vertex = Outlet). Properties are the D10 per-feature fields; hydrata fills
the defaults, and this module fills them again so a hand-built package (or a
map-editor insert that sent explicit NULLs) still gets the documented values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from run_anuga.culverts import constants


class CulvertBuildError(ValueError):
    """A culvert cannot be built into the model; the message says why, in plain words."""


#: The multi-process refusal (raised by run_sim and by the coupler, TASK-3642). It starts with
#: the stable "Cannot build the culverts:" prefix that hydrata's Run.user_message passes
#: through verbatim (TASK-3739); without it the user saw the generic failure sentence.
MPI_REFUSAL_MESSAGE = (
    "Cannot build the culverts: culverts are not supported in multi-process (MPI) runs yet; "
    "run this scenario on a single process."
)


@dataclass
class Culvert:
    feature_id: object
    name: str
    shape: str
    coordinates: list
    barrels: int = constants.DEFAULT_BARRELS
    wall_thickness_m: float = constants.DEFAULT_WALL_THICKNESS_M
    blockage_pct: float = constants.DEFAULT_BLOCKAGE_PCT
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    diameter_m: Optional[float] = None
    upstream_invert_m: Optional[float] = None
    downstream_invert_m: Optional[float] = None
    entrance_type: Optional[str] = None
    material: Optional[str] = None
    manning_n: Optional[float] = None
    length_override_m: Optional[float] = None
    description: Optional[str] = None

    @property
    def label(self) -> str:
        return f"Culvert '{self.name}' (feature {self.feature_id})"

    @property
    def barrel_width_m(self) -> float:
        return float(self.diameter_m if self.shape == "circular" else self.width_m)

    @property
    def rise_m(self) -> float:
        return float(self.diameter_m if self.shape == "circular" else self.height_m)

    @property
    def footprint_width_m(self) -> float:
        """barrels * width + (barrels + 1) * wall_thickness (D10)."""
        return self.barrels * self.barrel_width_m + (self.barrels + 1) * self.wall_thickness_m

    @property
    def inlet_xy(self) -> tuple:
        return tuple(self.coordinates[0])

    @property
    def outlet_xy(self) -> tuple:
        return tuple(self.coordinates[-1])

    @property
    def drawn_length_m(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.coordinates, self.coordinates[1:]))

    @property
    def length_m(self) -> float:
        return float(self.length_override_m) if self.length_override_m else self.drawn_length_m

    def outward_direction(self, end: str) -> tuple:
        """Unit vector pointing AWAY from the barrel at ``end`` ('inlet'|'outlet').

        Taken from the end segment, so a bent polyline still points the right way.
        """
        if end == "inlet":
            tip, neighbour = self.coordinates[0], self.coordinates[1]
        else:
            tip, neighbour = self.coordinates[-1], self.coordinates[-2]
        dx, dy = tip[0] - neighbour[0], tip[1] - neighbour[1]
        norm = math.hypot(dx, dy)
        return dx / norm, dy / norm


def _number(value, name, problems, *, integer=False):
    if value is None or value == "":
        return None
    try:
        number = int(value) if integer and float(value) == int(float(value)) else float(value)
    except (TypeError, ValueError):
        problems.append(f"{name} must be a number, got {value!r}")
        return None
    if integer and not isinstance(number, int):
        problems.append(f"{name} must be a whole number, got {value!r}")
        return None
    if isinstance(number, float) and not math.isfinite(number):
        problems.append(f"{name} must be a finite number, got {value!r}")
        return None
    return number


def _line_coordinates(geometry):
    """The single line's [x, y] vertices, or None when it is not one line."""
    geometry = geometry or {}
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if kind == "MultiLineString":
        if len(coordinates) != 1:
            return None
        coordinates = coordinates[0]
    elif kind != "LineString":
        return None
    try:
        points = [[float(p[0]), float(p[1])] for p in coordinates]
    except (TypeError, ValueError, IndexError):
        return None
    # Drop repeated consecutive vertices (a double-click on the map).
    deduped = [points[0]] if points else []
    for point in points[1:]:
        if point != deduped[-1]:
            deduped.append(point)
    return deduped


def _parse_feature(index, feature):
    """One Culvert, plus the list of plain-English problems found."""
    problems = []
    properties = dict((feature or {}).get("properties") or {})
    feature_id = (feature or {}).get("id")
    if feature_id is None:
        feature_id = properties.get("fid", f"#{index + 1}")
    name = properties.get("name") or f"Culvert {feature_id}"

    coordinates = _line_coordinates((feature or {}).get("geometry"))
    if not coordinates or len(coordinates) < 2:
        problems.append("its geometry must be a line from the Inlet to the Outlet with two distinct points")
        coordinates = None

    shape = (properties.get("shape") or "").strip().lower() or None
    width = _number(properties.get("width_m"), "width_m", problems)
    height = _number(properties.get("height_m"), "height_m", problems)
    diameter = _number(properties.get("diameter_m"), "diameter_m", problems)
    if shape is None:
        problems.append("shape is missing (box or circular)")
    elif shape not in constants.SHAPES:
        problems.append(f"shape must be box or circular, got {shape!r}")
    elif shape == "box" and not (width and width > 0 and height and height > 0):
        problems.append("a box culvert needs width_m and height_m greater than 0")
    elif shape == "circular" and not (diameter and diameter > 0):
        problems.append("a circular culvert needs diameter_m greater than 0")

    barrels = _number(properties.get("barrels"), "barrels", problems, integer=True)
    if barrels is None:
        barrels = constants.DEFAULT_BARRELS
    elif not 1 <= barrels <= constants.MAX_BARRELS:
        problems.append(f"barrels must be between 1 and {constants.MAX_BARRELS}, got {barrels}")

    wall = _number(properties.get("wall_thickness_m"), "wall_thickness_m", problems)
    if wall is None:
        wall = constants.DEFAULT_WALL_THICKNESS_M
    elif wall < 0:
        problems.append(f"wall_thickness_m cannot be negative, got {wall}")

    blockage = _number(properties.get("blockage_pct"), "blockage_pct", problems)
    if blockage is None:
        blockage = constants.DEFAULT_BLOCKAGE_PCT
    elif not 0 <= blockage <= 100:
        problems.append(f"blockage_pct must be between 0 and 100, got {blockage}")

    length_override = _number(properties.get("length_override_m"), "length_override_m", problems)
    if length_override is not None and length_override <= 0:
        problems.append(f"length_override_m must be greater than 0 when set, got {length_override}")

    culvert = Culvert(
        feature_id=feature_id,
        name=str(name),
        shape=shape or "",
        coordinates=coordinates or [],
        barrels=barrels,
        wall_thickness_m=wall,
        blockage_pct=blockage,
        width_m=width,
        height_m=height,
        diameter_m=diameter,
        upstream_invert_m=_number(properties.get("upstream_invert_m"), "upstream_invert_m", problems),
        downstream_invert_m=_number(properties.get("downstream_invert_m"), "downstream_invert_m", problems),
        entrance_type=properties.get("entrance_type"),
        material=properties.get("material"),
        manning_n=_number(properties.get("manning_n"), "manning_n", problems),
        length_override_m=length_override,
        description=properties.get("description"),
    )
    return culvert, problems


def load_culverts(culvert_geojson) -> list:
    """Parse a culvert FeatureCollection into Culvert records.

    Returns [] for a missing layer. Raises CulvertBuildError naming every bad
    feature (all of them at once, so the user fixes the layer in one pass).
    """
    if not culvert_geojson:
        return []
    culverts, failures = [], []
    for index, feature in enumerate(culvert_geojson.get("features") or []):
        culvert, problems = _parse_feature(index, feature)
        if problems:
            failures.append(f"{culvert.label}: " + "; ".join(problems))
        else:
            culverts.append(culvert)
    if failures:
        raise CulvertBuildError("Cannot build the culverts:\n  " + "\n  ".join(failures))
    return culverts
