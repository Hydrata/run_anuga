#!/usr/bin/env python3
"""
Prepare the Merewether urban flood benchmark scenario for run_anuga.

Reads source data from anuga_core validation tests and writes processed files to
examples/merewether/inputs/. Run once; outputs are committed to the repo.

Source: ARR Project 15 Merewether benchmark (Smith, Rahman & Wasko 2016)
        June 2007 "Pasha Bulker" storm, Newcastle NSW
CRS: EPSG:32756 (WGS84 UTM Zone 56S)

Usage:
    python scripts/prepare_merewether_scenario.py
"""

import csv
import json
import subprocess
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ANUGA_MEREWETHER = Path(
    "/home/dave/hydrata/anuga_core/validation_tests/case_studies/merewether"
)
OUT_DIR = Path("examples/merewether/inputs")
TMP_DIR = Path("/tmp/merewether_dem")

# Elevation added to building footprints in the DEM.
# Matches the original ANUGA benchmark (house_height=3.0 in runMerewether.py).
# Makes building interiors dry → max_speed≈0 → no CFL constraint → no instability.
BURN_HEIGHT_M = 3.0

# House files used by the benchmark (matches project.py holes list).
# Results in 57 polygon features, not 59 numbered houses:
#   house030.csv is excluded (commented out in project.py — it overlaps others).
#   house032_033.csv is used instead of the individual house032/house033 CSVs.
HOUSE_FILES = (
    [f"house{i:03d}.csv" for i in range(30)]  # 000–029
    + ["house031.csv", "house032_033.csv"]
    + [f"house{i:03d}.csv" for i in range(34, 59)]  # 034–058
)


def csv_to_coords(path: Path) -> list[list[float]]:
    """Read a two-column CSV (no header) and return list of [x, y] pairs."""
    coords = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            row = [c.strip() for c in row if c.strip()]
            if len(row) >= 2:
                try:
                    coords.append([float(row[0]), float(row[1])])
                except ValueError:
                    pass  # skip header rows if any
    return coords


def close_ring(coords: list[list[float]]) -> list[list[float]]:
    """Ensure the coordinate ring is closed (first == last)."""
    if coords and coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    return coords


# ---------------------------------------------------------------------------
# Step 1: DEM — convert topography1.asc → dem.tif
# ---------------------------------------------------------------------------
def make_dem():
    print("Converting DEM (topography1.asc → dem.tif)…")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    asc_path = TMP_DIR / "topography1.asc"

    with zipfile.ZipFile(ANUGA_MEREWETHER / "topography1.zip") as z:
        z.extract("topography1.asc", TMP_DIR)

    out_tif = OUT_DIR / "dem.tif"
    subprocess.run(
        [
            "gdal_translate",
            "-of", "GTiff",
            "-a_srs", "EPSG:32756",
            str(asc_path),
            str(out_tif),
        ],
        check=True,
    )
    print(f"  Written: {out_tif}")


# ---------------------------------------------------------------------------
# Step 1b: burn building footprints into dem.tif
# ---------------------------------------------------------------------------
def burn_buildings_into_dem():
    """Burn building footprints (+BURN_HEIGHT_M) into dem.tif using gdal_rasterize."""
    structure_path = OUT_DIR / "structure.geojson"
    dem_path = OUT_DIR / "dem.tif"

    with open(structure_path) as f:
        structures = json.load(f)

    n_shapes = sum(1 for feat in structures["features"] if feat.get("geometry"))
    if not n_shapes:
        print("  No building shapes to burn.")
        return

    subprocess.run(
        [
            "gdal_rasterize",
            "-add",
            "-burn", str(BURN_HEIGHT_M),
            str(structure_path),
            str(dem_path),
        ],
        check=True,
    )
    print(f"  Burned {n_shapes} buildings (+{BURN_HEIGHT_M}m) into dem.tif")


# ---------------------------------------------------------------------------
# Step 1c: clear structure.geojson after burning
# ---------------------------------------------------------------------------
def clear_structure_methods():
    """Remove buildings from structure.geojson — elevation burn replaces them.
    Prevents n=10 Manning's friction being applied on top of elevation bumps."""
    out = OUT_DIR / "structure.geojson"
    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:32756"}},
        "features": [],
    }
    out.write_text(json.dumps(geojson, indent=2))
    print("  Cleared structure.geojson (buildings encoded in DEM elevation burn)")


# ---------------------------------------------------------------------------
# Step 2: boundary.geojson
# Bounding box from extent.csv: SW (382250, 6354265) → NE (382571, 6354681)
# Each side is a 2m inset LineString with boundary type matching ANUGA benchmark.
# ---------------------------------------------------------------------------
def make_boundary():
    print("Creating boundary.geojson…")
    # Original corners (from extent.csv)
    x_min, y_min = 382250.0, 6354265.0
    x_max, y_max = 382571.0, 6354681.0
    inset = 2.0

    # Inset corners
    x1, y1 = x_min + inset, y_min + inset
    x2, y2 = x_max - inset, y_max - inset

    features = [
        {
            "type": "Feature",
            "id": "south",
            "properties": {"boundary": "Reflective", "location": "External"},
            "geometry": {
                "type": "LineString",
                # bottom edge: left → right
                "coordinates": [[x1, y1], [x2, y1]],
            },
        },
        {
            "type": "Feature",
            "id": "east",
            "properties": {"boundary": "Transmissive", "location": "External"},
            "geometry": {
                "type": "LineString",
                # right edge: bottom → top (outflow)
                "coordinates": [[x2, y1], [x2, y2]],
            },
        },
        {
            "type": "Feature",
            "id": "north",
            "properties": {"boundary": "Transmissive", "location": "External"},
            "geometry": {
                "type": "LineString",
                # top edge: right → left (outflow)
                "coordinates": [[x2, y2], [x1, y2]],
            },
        },
        {
            "type": "Feature",
            "id": "west",
            "properties": {"boundary": "Reflective", "location": "External"},
            "geometry": {
                "type": "LineString",
                # left edge: top → bottom
                "coordinates": [[x1, y2], [x1, y1]],
            },
        },
    ]

    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:32756"}},
        "features": features,
    }
    out = OUT_DIR / "boundary.geojson"
    out.write_text(json.dumps(geojson, indent=2))
    print(f"  Written: {out}")


# ---------------------------------------------------------------------------
# Step 3: inflow.geojson
# Fixed discharge via Inlet_operator (Surface LineString, 19.7 m³/s).
# Geometry matches the inlet region used in runMerewether.py:
#   center=(382265.0, 6354280.0), radius=10.0
# ---------------------------------------------------------------------------
def make_inflow():
    print("Creating inflow.geojson…")
    # LineString through the inlet centre — matches runMerewether.py line0
    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:32756"}},
        "features": [
            {
                "type": "Feature",
                "id": "inlet",
                "properties": {
                    "type": "Surface",
                    "data": "19.7",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[382255.0, 6354280.0], [382275.0, 6354280.0]],
                },
            }
        ],
    }
    out = OUT_DIR / "inflow.geojson"
    out.write_text(json.dumps(geojson, indent=2))
    print(f"  Written: {out}")


# ---------------------------------------------------------------------------
# Step 4: friction.geojson
# RoadPolygon.csv → GeoJSON Polygon with mannings=0.02.
# ---------------------------------------------------------------------------
def make_friction():
    print("Creating friction.geojson…")
    road_csv = ANUGA_MEREWETHER / "Road" / "RoadPolygon.csv"
    coords = csv_to_coords(road_csv)
    coords = close_ring(coords)

    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:32756"}},
        "features": [
            {
                "type": "Feature",
                "id": "roads",
                "properties": {"mannings": 0.02},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
            }
        ],
    }
    out = OUT_DIR / "friction.geojson"
    out.write_text(json.dumps(geojson, indent=2))
    print(f"  Written: {out}  ({len(coords)-1} road polygon vertices)")


# ---------------------------------------------------------------------------
# Step 5: structure.geojson
# 59 building footprints → GeoJSON FeatureCollection, method=Mannings.
# ---------------------------------------------------------------------------
def make_structures():
    print("Creating structure.geojson…")
    houses_dir = ANUGA_MEREWETHER / "houses"
    features = []
    for fname in HOUSE_FILES:
        path = houses_dir / fname
        coords = csv_to_coords(path)
        coords = close_ring(coords)
        name = Path(fname).stem
        features.append(
            {
                "type": "Feature",
                "id": name,
                "properties": {"method": "Mannings"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
            }
        )

    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:32756"}},
        "features": features,
    }
    out = OUT_DIR / "structure.geojson"
    out.write_text(json.dumps(geojson, indent=2))
    print(f"  Written: {out}  ({len(features)} buildings)")


# ---------------------------------------------------------------------------
# Step 6: validation/observation_points.geojson
# From Observations/ObservationPoints.csv
# ---------------------------------------------------------------------------
def make_observation_points():
    print("Creating validation/observation_points.geojson…")
    obs_csv = ANUGA_MEREWETHER / "Observations" / "ObservationPoints.csv"

    features = []
    with open(obs_csv) as f:
        reader = csv.DictReader(f)
        # Strip leading/trailing whitespace from all column names
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            x = float(row["x"])
            y = float(row["y"])
            obs_id = int(row["ID"])
            field_stage = float(row["stage (Draft Report)"])
            arr_stage = float(row["stage (ARR Report Final)"])
            tuflow_stage = float(row["TUFLOW (ARR Report Final)"])
            features.append(
                {
                    "type": "Feature",
                    "id": str(obs_id),
                    "properties": {
                        "id": obs_id,
                        "field_stage_m": field_stage,
                        "arr_report_m": arr_stage,
                        "tuflow_m": tuflow_stage,
                        "tolerance_m": 0.3,
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [x, y],
                    },
                }
            )

    # Sort by ID for consistent output
    features.sort(key=lambda f: f["properties"]["id"])

    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:32756"}},
        "features": features,
    }
    val_dir = Path("examples/merewether/validation")
    val_dir.mkdir(parents=True, exist_ok=True)
    out = val_dir / "observation_points.geojson"
    out.write_text(json.dumps(geojson, indent=2))
    print(f"  Written: {out}  ({len(features)} points)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    make_dem()
    make_structures()          # writes structure.geojson WITH building polygons
    burn_buildings_into_dem()  # burns them into dem.tif
    clear_structure_methods()  # empties structure.geojson (no n=10 friction on bumps)
    make_friction()
    make_inflow()
    make_boundary()
    make_observation_points()

    print("\nDone. All inputs written to examples/merewether/")


if __name__ == "__main__":
    main()
