import argparse
import datetime
import glob
import json
import logging
import logging.handlers
import math
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from run_anuga._imports import import_optional
from run_anuga import defaults
from run_anuga._logging import install_mname_filter
from run_anuga.config import ScenarioConfig

try:
    from celery.utils.log import get_task_logger
    logger = get_task_logger(__name__)
    from django.conf import settings
except ImportError:
    logger = logging.getLogger(__name__)
    settings = dict()

# Stamp anuga_core's mname/lnum record fields so run_anuga logs format cleanly
# when they propagate to anuga's root %(mname)s formatter (TASK-1276).
install_mname_filter(logger)


@dataclass
class RunContext:
    """Typed replacement for the (package_dir, username, password) tuple."""
    package_dir: str
    username: Optional[str] = None
    password: Optional[str] = None
    # Cached at run_sim startup so _report_run_error reuses the same config
    # the run started with even if scenario.json is overwritten mid-run.
    scenario_config: Optional[dict] = None


def is_dir_check(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def _load_package_data(package_dir):
    """Load and validate scenario config + input files. Pure Python, no geo deps."""
    if not os.path.isfile(os.path.join(package_dir, 'scenario.json')):
        raise FileNotFoundError(f'Could not find "scenario.json" in {package_dir}')

    input_data = dict()
    with open(os.path.join(package_dir, 'scenario.json')) as f:
        input_data['scenario_config'] = json.load(f)
    try:
        ScenarioConfig.model_validate(input_data['scenario_config'])
    except Exception as e:
        logger.warning(f"Scenario validation: {e}")
    project_id = input_data['scenario_config'].get('project')
    scenario_id = input_data['scenario_config'].get('id')
    run_id = input_data['scenario_config'].get('run_id')
    input_data['run_label'] = f"run_{project_id}_{scenario_id}_{run_id}"
    input_data['output_directory'] = os.path.join(package_dir, f'outputs_{project_id}_{scenario_id}_{run_id}')
    input_data['mesh_filepath'] = f"{input_data['output_directory']}/run_{project_id}_{scenario_id}_{run_id}.msh"
    Path(input_data['output_directory']).mkdir(parents=True, exist_ok=True)
    input_data['checkpoint_directory'] = f"{input_data['output_directory']}/checkpoints/"
    Path(input_data['checkpoint_directory']).mkdir(parents=True, exist_ok=True)

    input_data['boundary_filename'] = os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get('boundary')}")
    with open(input_data['boundary_filename']) as f:
        input_data['boundary'] = json.load(f)

    data_types = [
        'friction',
        'inflow',
        'rainfall',
        'structure',
        'mesh_region',
        'network',
        'catchment',
        'nodes',
        'links'
    ]
    for data_type in data_types:
        filepath = os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get(data_type)}")
        if input_data['scenario_config'].get(data_type) and os.path.isfile(filepath):
            input_data[f'{data_type}_filename'] = filepath
            with open(filepath) as f:
                input_data[data_type] = json.load(f)

    elevation_filepath = os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get('elevation')}")
    if input_data['scenario_config'].get('elevation') and os.path.isfile(elevation_filepath):
        input_data['elevation_filename'] = elevation_filepath

    friction_raster_filepath = os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get('friction_raster')}")
    if input_data['scenario_config'].get('friction_raster') and os.path.isfile(friction_raster_filepath):
        input_data['friction_raster_filename'] = friction_raster_filepath

    if input_data['scenario_config'].get('resolution'):
        input_data['resolution'] = input_data['scenario_config'].get('resolution')

    return input_data


def setup_input_data(package_dir):
    """Full setup — requires geo deps for boundary processing."""
    input_data = _load_package_data(package_dir)

    if len(input_data['boundary'].get('features')) == 0:
        raise AttributeError('No boundary features found')
    boundary_polygon, boundary_tags = create_boundary_polygon_from_boundaries(
        input_data['boundary']
    )
    if len(boundary_polygon) == 0 or len(boundary_tags) == 0:
        raise AttributeError('No boundary data found')
    input_data['boundary_polygon'] = boundary_polygon
    input_data['boundary_tags'] = boundary_tags
    return input_data


def update_web_interface(run_args, data, files=None):
    package_dir, username, password = run_args.package_dir, run_args.username, run_args.password
    if username and password:
        requests = import_optional("requests")
        from run_anuga._http import post_to_control_server

        input_data = setup_input_data(package_dir)
        data['project'] = input_data['scenario_config'].get('project')
        data['scenario'] = input_data['scenario_config'].get('id')
        run_id = input_data['scenario_config'].get('run_id')
        control_server = input_data['scenario_config'].get('control_server')
        url = f"{control_server}anuga/api/{data['project']}/{data['scenario']}/run/{run_id}/"
        auth = requests.auth.HTTPBasicAuth(username, password)
        # logger.critical(f"hydrata.com post:{data}")
        post_to_control_server(url, auth=auth, method="PATCH", data=data, files=files)



def get_utm_geo_reference(epsg_str):
    """
    TASK-1260: Derive an anuga.Geo_reference from an EPSG string using pyproj.

    Replaces the fragile ``int(epsg[-2:])`` slice (e.g. "EPSG:32601"[-2:] →
    "01" → zone=1 works, but edge cases like 3-digit zones could fail).
    pyproj reliably extracts the UTM zone number for both pure-UTM (32655/32755)
    and projected CRS whose name includes 'zone NN' (28355 / MGA).

    Parameters
    ----------
    epsg_str : str
        EPSG code, either "EPSG:32755" or bare "32755".

    Returns
    -------
    anuga.Geo_reference
        With ``zone`` set to the correct UTM zone integer.
    """
    import re
    anuga = import_optional("anuga")
    pyproj = import_optional("pyproj")
    CRS = pyproj.CRS

    # Normalise to integer EPSG code
    epsg_clean = epsg_str.strip()
    if ':' in epsg_clean:
        epsg_int = int(epsg_clean.split(':')[-1])
    else:
        epsg_int = int(epsg_clean)

    crs = CRS.from_epsg(epsg_int)

    # pyproj.CRS.utm_zone returns e.g. "55N", "55S", "1N", or None for
    # projected CRS that are not pure UTM (e.g. GDA94/MGA).
    utm_zone_str = crs.utm_zone
    if utm_zone_str:
        zone = int(re.search(r'\d+', utm_zone_str).group())
    else:
        # Fall back to parsing the CRS name (e.g. "GDA94 / MGA zone 55")
        m = re.search(r'zone\s+(\d+)', crs.name, re.IGNORECASE)
        if not m:
            raise ValueError(
                f"Cannot determine UTM zone from EPSG:{epsg_int} "
                f"(name={crs.name!r}, utm_zone={utm_zone_str!r}). "
                "Provide a UTM or MGA projected CRS."
            )
        zone = int(m.group(1))

    return anuga.Geo_reference(zone=zone)


def create_anuga_mesh(input_data):
    anuga = import_optional("anuga")
    Geo_reference = anuga.Geo_reference
    mesh_filepath = input_data['mesh_filepath']
    triangle_resolution = (input_data['scenario_config'].get('resolution') ** 2) / 2
    interior_regions = make_interior_regions(input_data)
    # interior_holes, hole_tags = make_interior_holes_and_tags(input_data)
    bounding_polygon = input_data['boundary_polygon']
    boundary_tags = input_data['boundary_tags']
    logger.critical("creating anuga_mesh")
    if input_data.get('structure_filename'):
        burn_structures_into_raster = subprocess.run([
            "gdal_rasterize",
            "-burn", str(defaults.BUILDING_BURN_HEIGHT_M), "-add",
            input_data['structure_filename'],
            input_data['elevation_filename']
        ],
            capture_output=True,
            universal_newlines=True
        )
        logger.critical(burn_structures_into_raster.stdout)
        if burn_structures_into_raster.returncode != 0:
            logger.critical(burn_structures_into_raster.stderr)
            raise UserWarning(burn_structures_into_raster.stderr)
    mesh_geo_reference = get_utm_geo_reference(input_data['scenario_config'].get('epsg'))
    anuga_mesh = anuga.pmesh.mesh_interface.create_mesh_from_regions(
        bounding_polygon=bounding_polygon,
        boundary_tags=boundary_tags,
        maximum_triangle_area=triangle_resolution,
        interior_regions=interior_regions,
        # interior_holes=interior_holes,
        mesh_geo_reference=mesh_geo_reference,
        # hole_tags=hole_tags,
        filename=mesh_filepath,
        use_cache=False,
        verbose=False,
        fail_if_polygons_outside=False
    )
    logger.critical(f"mesh_triangle_count={len(anuga_mesh.tri_mesh.triangles)}")
    return mesh_filepath, anuga_mesh


def get_sql_triangles_from_anuga_mesh(anuga_mesh):
    vertices = anuga_mesh.tri_mesh.vertices
    triangles = anuga_mesh.tri_mesh.triangles
    output = "MULTIPOLYGON ("
    for triangle in triangles:
        # get coordinates:
        one = str(vertices[triangle[0]])[2:-1]
        two = str(vertices[triangle[1]])[2:-1]
        three = str(vertices[triangle[2]])[2:-1]
        four = str(vertices[triangle[0]])[2:-1]
        triangle_string = f"(({one},{two},{three},{four})),"
        output += str(triangle_string)
    output = output[:-1] + ")"
    return output


def make_interior_regions(input_data):
    interior_regions = list()
    if input_data.get('mesh_region'):
        for mesh_region in input_data['mesh_region']['features']:
            mesh_polygon = _extract_polygon_outer_ring(mesh_region.get('geometry'))
            mesh_resolution = mesh_region.get('properties').get('resolution')
            interior_regions.append((mesh_polygon, mesh_resolution,))
    return interior_regions


def make_interior_holes_and_tags(input_data):
    interior_holes = list()
    hole_tags = list()
    if input_data.get('structure'):
        for structure in input_data['structure']['features']:
            if structure.get('properties').get('method') == 'Mannings':
                continue
            structure_polygon = _extract_polygon_outer_ring(structure.get('geometry'))
            interior_holes.append(structure_polygon)
            if structure.get('properties').get('method') == 'Holes':
                hole_tags.append(None)
            elif structure.get('properties').get('method') == 'Reflective':
                hole_tags.append({'reflective': [i for i in range(len(structure_polygon))]})
            else:
                logger.error(f"Unknown interior hole type found: {structure.get('properties').get('method')}")
    if len(interior_holes) == 0:
        interior_holes = None
        hole_tags = None
    return interior_holes, hole_tags


def make_frictions(input_data):
    # Raster precedence (TASK-830 / TASK-1259): a friction raster sets the
    # base Manning's value at full raster resolution across the whole domain.
    # Per-structure Manning's-n patches (method='Mannings') must STILL overlay
    # the raster — ANUGA's composite_quantity_setting_function evaluates entries
    # in order, last-match wins, so appending the structure patches AFTER the
    # raster entry applies them on top.
    #
    # Old behaviour (pre-TASK-1259): early-return dropped all structure patches
    # when a raster was present.  New behaviour: raster first, then structure
    # patches, no 'All' fallback (raster already covers the whole domain).
    #
    # Without a raster: polygon-only path unchanged (structure + friction polys
    # + 'All' fallback).
    # See docs/reports/2026-05-13-q-1-task-830-friction-raster-attachment.html.
    frictions = list()
    if input_data.get('friction_raster_filename'):
        frictions.append(['Extent', input_data['friction_raster_filename']])
        # Overlay per-structure Manning's-n patches on top of the raster.
        if input_data.get('structure'):
            for structure in input_data['structure']['features']:
                if structure.get('properties').get('method') == 'Mannings':
                    structure_polygon = _extract_polygon_outer_ring(structure.get('geometry'))
                    frictions.append((structure_polygon, defaults.BUILDING_MANNINGS_N,))
        return frictions
    if input_data.get('structure'):
        for structure in input_data['structure']['features']:
            if structure.get('properties').get('method') == 'Mannings':
                structure_polygon = _extract_polygon_outer_ring(structure.get('geometry'))
                frictions.append((structure_polygon, defaults.BUILDING_MANNINGS_N,))
    if input_data.get('friction'):
        for friction in input_data['friction']['features']:
            friction_polygon = _extract_polygon_outer_ring(friction.get('geometry'))
            friction_value = friction.get('properties').get('mannings')
            frictions.append((friction_polygon, friction_value,))
    frictions.append(['All', defaults.DEFAULT_MANNINGS_N])
    return frictions


def assert_raster_has_no_nodata_inside_boundary(raster_path, boundary_polygon, *, quantity_name):
    """Pre-flight guard: raise a clear error if a raster has nodata cells inside the model boundary.

    ANUGA seats elevation/friction values onto mesh centroids via
    ``composite_quantity_setting_function(..., nan_treatment='exception')``. If
    any sample point lands on a nodata cell, ANUGA raises an opaque exception
    deep inside ``composite_quantity_setting_function`` mid-build, with no hint
    that the cause is a data gap in the input raster. This function runs the
    same check up-front, on rank 0, before the set_quantity calls, and raises a
    message that names the raster, the count of offending cells, and the fix.

    We deliberately KEEP ``nan_treatment='exception'`` everywhere (we never want
    to silently fabricate a bed/friction value); this guard simply surfaces the
    failure earlier and more clearly.

    Nodata detection: a cell is "nodata" if it equals the raster's declared
    nodata tag (e.g. the TASK-1136 standard ``-9999``) OR is NaN. Both are
    handled because GeoTIFFs in the wild use either convention. If the raster
    declares NO nodata value, there is nothing to check and we return (pass) —
    a stray NaN with no declared nodata is left for ANUGA to surface.

    Inside-boundary test: the boundary polygon is rasterised onto the raster's
    own grid/transform via ``rasterio.features.geometry_mask`` (vectorised, one
    pass — far cheaper than a per-cell shapely point-in-polygon test), then
    intersected with the nodata mask.

    Parameters
    ----------
    raster_path : str
        Path to the elevation or friction GeoTIFF (in the raster's projected CRS).
    boundary_polygon : list
        Flat list of ``[x, y]`` vertices for the model boundary, in the raster's
        projected CRS (``input_data['boundary_polygon']``). A no-op if empty.
    quantity_name : str
        Keyword-only label ('elevation' or 'friction') used in the error message.

    Raises
    ------
    ValueError
        If one or more nodata cells fall inside the model boundary.
    """
    if not boundary_polygon:
        # No boundary to test against — nothing this guard can assert.
        return
    rasterio = import_optional("rasterio")
    features = import_optional("rasterio.features")
    numpy = import_optional("numpy")

    with rasterio.open(raster_path) as dataset:
        band = dataset.read(1)
        nodata_value = dataset.nodata
        transform = dataset.transform
        out_shape = (dataset.height, dataset.width)

    # NaN is always treated as nodata. A finite declared nodata tag (e.g. -9999)
    # is matched exactly. If neither applies, there is nothing to flag.
    nodata_mask = numpy.isnan(band)
    if nodata_value is not None and not numpy.isnan(nodata_value):
        nodata_mask = nodata_mask | (band == nodata_value)
    # No nodata cells (no declared tag and no NaN, or a declared tag that no
    # cell matches) — nothing to check.
    if not nodata_mask.any():
        return

    # Close the ring so shapely/rasterio treats it as a polygon, not a line.
    ring = [tuple(point) for point in boundary_polygon]
    if ring[0] != ring[-1]:
        ring = ring + [ring[0]]
    boundary_geometry = {'type': 'Polygon', 'coordinates': [ring]}

    # geometry_mask returns True OUTSIDE the geometry by default (invert=False),
    # so cells inside the boundary are where the mask is False.
    outside_mask = features.geometry_mask(
        [boundary_geometry],
        out_shape=out_shape,
        transform=transform,
        invert=False,
        all_touched=True,
    )
    inside_mask = ~outside_mask
    offending = nodata_mask & inside_mask
    offending_count = int(offending.sum())
    if offending_count > 0:
        raise ValueError(
            f"{quantity_name} raster '{raster_path}' has {offending_count} nodata "
            f"cells inside the model boundary; fill the data gaps "
            f"(e.g. gdal_fillnodata) or reselect the terrain extent. "
            f"ANUGA cannot seat a bed/friction value there."
        )


def correction_for_polar_quadrants(base, height):
    result = 0
    result = 0 if base > 0 and height > 0 else result
    result = math.pi if base < 0 and height > 0 else result
    result = math.pi if base < 0 and height < 0 else result
    result = 2 * math.pi if base > 0 and height < 0 else result
    return result


def lookup_boundary_tag(index, boundary_tags):
    for key in boundary_tags.keys():
        if index in boundary_tags[key]:
            return key


def _flatten_line_coordinates(geometry):
    """
    Return a flat list of [x, y] points from a LineString or MultiLineString
    GeoJSON geometry. PostGIS / GeoServer normalises every boundary feature
    to MultiLineString regardless of input — even when the source file is
    LineString — so this helper has to handle both. For MultiLineString with
    a single ring we flatten one level; for true multi-rings we concatenate
    (the boundary polygon is sorted clockwise by feature centroid afterwards,
    so the join order within a feature does not matter for sorting).

    Returns [] for missing or empty coordinates; the debug log surfaces
    data-quality issues without aborting the simulation.
    """
    raw_coords = geometry.get('coordinates')
    coords = raw_coords or []
    gtype = geometry.get('type')
    if not coords:
        logger.debug(
            "_flatten_line_coordinates: empty/missing coordinates "
            "(type=%r, raw=%r); returning []",
            gtype, raw_coords,
        )
        return []
    if gtype == 'MultiLineString':
        return [pt for line in coords for pt in line]
    if gtype != 'LineString':
        logger.debug(
            "_flatten_line_coordinates: unsupported geometry type %r; "
            "returning coordinates as-is",
            gtype,
        )
    return coords


def _extract_polygon_outer_ring(geometry):
    """
    Return the 2-D outer ring of a Polygon or MultiPolygon GeoJSON geometry.
    The BE can serialise the same polygon-shaped feature as either Polygon
    or MultiPolygon depending on storage and round-trip path, so ANUGA-side
    callers (set_quantity, Polygonal_rate_operator, shapely.Polygon,
    gdalwarp -cutline) must accept both shapes — they all want an (N, 2)
    list of vertices, not a (1, N, 2) list-of-polygons.
    For MultiPolygon with multiple sub-polygons, the first sub-polygon's
    outer ring is returned and a warning is logged. Inner rings (holes)
    inside the first sub-polygon are also dropped, matching the previous
    Polygon-only behaviour which only ever consumed coordinates[0].

    Returns [] for missing or empty coordinates; the debug log surfaces
    data-quality issues without aborting the simulation.
    """
    raw_coords = geometry.get('coordinates')
    coords = raw_coords or []
    gtype = geometry.get('type')
    if not coords:
        logger.debug(
            "_extract_polygon_outer_ring: empty/missing coordinates "
            "(type=%r, raw=%r); returning []",
            gtype, raw_coords,
        )
        return []
    if gtype == 'MultiPolygon':
        if len(coords) > 1:
            logger.warning(
                "MultiPolygon with %d sub-polygons; only the first will be used",
                len(coords),
            )
        return coords[0][0] if coords[0] else []
    if gtype != 'Polygon':
        logger.debug(
            "_extract_polygon_outer_ring: unsupported geometry type %r; "
            "falling through to coords[0]",
            gtype,
        )
    return coords[0]


def create_boundary_polygon_from_boundaries(boundaries_geojson):
    ogr = import_optional("osgeo.ogr")
    geometry_collection = ogr.Geometry(ogr.wkbGeometryCollection)
    if boundaries_geojson.get('crs'):
        epsg_code = boundaries_geojson.get('crs').get('properties').get('name').split(':')[-1]
    else:
        return list(), dict()
    # Create a dict of the available boundary tags
    boundary_tag_labels = dict()
    all_x_coordinates = list()
    all_y_coordinates = list()
    for index, feature in enumerate(boundaries_geojson.get('features')):
        if feature.get('properties').get('location') != "External":
            continue
        boundary_tag_labels[feature.get('properties').get('boundary')] = []
        geometry = ogr.CreateGeometryFromJson(json.dumps(feature.get('geometry')))
        geometry_collection.AddGeometry(geometry)
        # Collect a list of the coordinates associated with each boundary tag:
        feature_coordinates = _flatten_line_coordinates(feature.get('geometry'))
        for coordinate in feature_coordinates:
            all_x_coordinates.append(coordinate[0])
            all_y_coordinates.append(coordinate[1])
    srs = ogr.osr.SpatialReference()
    epsg_integer = int(epsg_code.split(':')[1] if ':' in epsg_code else epsg_code)
    srs.ImportFromEPSG(epsg_integer)

    # Find the center of our project
    if not all_x_coordinates or not all_y_coordinates:
        raise ValueError(
            "create_boundary_polygon_from_boundaries: no valid External-location boundary coordinates found"
        )
    max_x = max(all_x_coordinates)
    max_y = max(all_y_coordinates)
    min_x = min(all_x_coordinates)
    min_y = min(all_y_coordinates)
    mid_x = max_x - (max_x - min_x) / 2
    mid_y = max_y - (max_y - min_y) / 2
    line_list = list()

    # Now create and sort the line_list of boundary lines in a clockwise direction around it
    for index, feature in enumerate(boundaries_geojson.get('features')):
        if feature.get('properties').get('location') != "External":
            # discard any internal boundaries from the boundary_polygon
            continue
        geometry = ogr.CreateGeometryFromJson(json.dumps(feature.get('geometry')))
        centroid = json.loads(geometry.Centroid().ExportToJson()).get('coordinates')
        base = centroid[0] - mid_x
        height = centroid[1] - mid_y
        # the angle in polar coordinates will sort our boundary lines into the correct order
        angle = math.atan2(height, base)
        line_list.append({
            "centroid": centroid,
            "boundary": feature.get('properties').get('boundary'),
            "id": feature.get('id'),
            "angle": angle,
            "coordinates": _flatten_line_coordinates(feature.get('geometry')),
        })
    line_list.sort(key=lambda line: line.get('angle'), reverse=True)

    # Now join all our lines in clockwise order and create the boundary tags object
    boundary_polygon = list()
    boundary_tags_list = list()
    counter = 0
    boundary_tags = deepcopy(boundary_tag_labels)
    for line in line_list:
        for coordinate in line.get("coordinates"):
            boundary_polygon.append(coordinate)
            boundary_tags[line.get("boundary")].append(counter)
            boundary_tags_list.append(lookup_boundary_tag(counter, boundary_tags))
            counter += 1

    # now sort the boundary_polygon points in clockwise order, in case those original lines were drawn with different
    # directions
    boundary_polygon_with_angle_data = list()
    sorted_boundary_polygon = list()
    sorted_boundary_tags = deepcopy(boundary_tag_labels)
    for index, point in enumerate(boundary_polygon):
        base = point[0] - mid_x
        height = point[1] - mid_y
        angle = math.atan2(height, base)
        boundary_polygon_with_angle_data.append({
            "point": point,
            "boundary": lookup_boundary_tag(index, boundary_tags),
            "angle": angle
        })
    boundary_polygon_with_angle_data.sort(key=lambda point_blob: point_blob.get('angle'), reverse=True)
    for index, point_blob in enumerate(boundary_polygon_with_angle_data):
        sorted_boundary_polygon.append(point_blob.get('point'))
        sorted_boundary_tags[point_blob.get('boundary')].append(index)

    # # Make a dump of the centroids geometry (for debugging only - not returned anywhere).
    # output_driver_centroids = ogr.GetDriverByName('GeoJSON')
    # filepath_geojson_driver_centroids = os.path.join(package_dir, f'outputs_{run_label.split("run_")[1]}', f'{run_label}_boundary_centroids.geojson')
    # output_data_source_centroids = output_driver_centroids.CreateDataSource(filepath_geojson_driver_centroids)
    # output_layer_centroids = output_data_source_centroids.CreateLayer(filepath_geojson_driver_centroids, srs, geom_type=ogr.wkbPolygon)
    # feature_definition_centroids = output_layer_centroids.GetLayerDefn()
    # field_definition_centroids_1 = ogr.FieldDefn('index', ogr.OFTReal)
    # output_layer_centroids.CreateField(field_definition_centroids_1)
    # field_definition_centroids_2 = ogr.FieldDefn('angle', ogr.OFTReal)
    # output_layer_centroids.CreateField(field_definition_centroids_2)
    # field_definition_centroids_3 = ogr.FieldDefn('id', ogr.OFTString)
    # field_definition_centroids_3.SetWidth(1000)
    # output_layer_centroids.CreateField(field_definition_centroids_3)
    # for index, line in enumerate(line_list):
    #     output_feature_centroids = ogr.Feature(feature_definition_centroids)
    #     centroid = line.get('centroid')
    #     point = ogr.Geometry(ogr.wkbPoint)
    #     point.AddPoint(centroid[0], centroid[1])
    #     output_feature_centroids.SetGeometry(point)
    #     output_feature_centroids.SetField('index', index)
    #     output_feature_centroids.SetField('angle', line.get('angle'))
    #     output_feature_centroids.SetField('id', line.get('id'))
    #     output_layer_centroids.CreateFeature(output_feature_centroids)
    #
    # # Make a dump of the boundary polygon geometry (for debugging only - not returned anywhere).
    # output_driver = ogr.GetDriverByName('GeoJSON')
    # filepath_geojson_driver = os.path.join(package_dir, f'outputs_{run_label.split("run_")[1]}', f'{run_label}_boundary_polygon.geojson')
    # output_data_source = output_driver.CreateDataSource(filepath_geojson_driver)
    # output_layer = output_data_source.CreateLayer(filepath_geojson_driver, srs, geom_type=ogr.wkbPolygon)
    # feature_definition = output_layer.GetLayerDefn()
    # field_definition_1 = ogr.FieldDefn('index', ogr.OFTReal)
    # output_layer.CreateField(field_definition_1)
    # field_definition_2 = ogr.FieldDefn('boundary', ogr.OFTString)
    # field_definition_2.SetWidth(1000)
    # output_layer_centroids.CreateField(field_definition_2)
    # for index, coordinate in enumerate(boundary_polygon):
    #     output_feature = ogr.Feature(feature_definition)
    #     point = ogr.Geometry(ogr.wkbPoint)
    #     point.AddPoint(coordinate[0], coordinate[1])
    #     output_feature.SetGeometry(point)
    #     output_feature.SetField('index', index)
    #     output_feature.SetField('boundary', boundary_tags_list[index])
    #     output_layer.CreateFeature(output_feature)

    return sorted_boundary_polygon, sorted_boundary_tags


def build_time_boundary_function(time_boundary_features, defaults=None):
    """Build the function passed to ``anuga.Time_boundary(domain, function=...)``.

    The Anuga Time boundary expects a callable ``f(t_seconds) -> [stage,
    xmom, ymom]`` returning conserved-quantity values for the model time t.

    Inputs:
        time_boundary_features: a list of GeoJSON Feature dicts with
            ``properties.boundary == 'Time'`` and a ``properties.data``
            value that has already been resolved server-side to either:
              * a numeric stage value (constant case), or
              * a list of ``{timestamp: ISO8601, value: number}`` dicts
                (timeseries case).

        defaults: the run_anuga.defaults module (or any object exposing
            float-coercible attributes). Currently unused but accepted for
            forward-compat — leaves room for momentum or factor defaults.

    Anuga's tag system collapses multiple Time-boundary features under a
    single 'Time' tag. We use the first feature's data and log a warning
    if there are multiple distinct Time boundaries — this is an authoring
    issue rather than a code limitation, and it surfaces clearly in logs.
    """
    if not time_boundary_features:
        raise ValueError(
            'build_time_boundary_function() called with no features'
        )
    if len(time_boundary_features) > 1:
        logger.warning(
            'Multiple Time boundary features supplied (%d); only the first '
            "will be applied (Anuga collapses them all under one 'Time' tag)",
            len(time_boundary_features),
        )

    first = time_boundary_features[0]
    data = (first.get('properties') or {}).get('data')

    # Constant case: a single numeric stage value applied for all t.
    if data is None:
        # Don't blow up at parse time — let Anuga surface the failure when
        # it tries to evaluate the function. Constant 0 is the safest default.
        logger.error(
            "Time boundary feature %s has no resolved data; defaulting to 0.0",
            first.get('id'),
        )
        return lambda t: [0.0, 0.0, 0.0]
    if isinstance(data, (int, float)):
        constant = float(data)
        return lambda t: [constant, 0.0, 0.0]
    if isinstance(data, str):
        # Shouldn't happen if Boundary.make_file did its job, but tolerate
        # a numeric string just in case.
        try:
            constant = float(data)
            return lambda t: [constant, 0.0, 0.0]
        except (TypeError, ValueError):
            raise ValueError(
                f'Time boundary data must be numeric or a list of '
                f'{{timestamp, value}} dicts, got string {data!r} that '
                f'could not be float-coerced'
            )

    # TimeSeries case: a list of {timestamp, value} dicts.
    if not isinstance(data, list) or not data:
        raise ValueError(
            f'Time boundary data must be numeric or a non-empty list of '
            f'{{timestamp, value}} dicts, got: {type(data).__name__}'
        )

    numpy = import_optional("numpy")
    pd = import_optional("pandas")

    timestamps = []
    values = []
    for row in data:
        ts = row.get('timestamp') if isinstance(row, dict) else None
        val = row.get('value') if isinstance(row, dict) else None
        if ts is None or val is None:
            raise ValueError(
                f'Time boundary data row missing timestamp or value: {row!r}'
            )
        timestamps.append(pd.to_datetime(ts, utc=True))
        values.append(float(val))

    # Convert timestamps → seconds since the first sample. The Anuga function
    # is called with model-time-in-seconds (starting at 0).
    timestamps_sec = numpy.array(
        [(t - timestamps[0]).total_seconds() for t in timestamps],
        dtype=float,
    )
    values_arr = numpy.array(values, dtype=float)

    def _time_function(t):
        # numpy.interp clamps below[0]→values[0] and above[-1]→values[-1],
        # which matches the Inflow timeseries semantics (forward-fill at edges).
        stage = float(numpy.interp(float(t), timestamps_sec, values_arr))
        return [stage, 0.0, 0.0]

    return _time_function


def apply_inflows_to_domain(
    input_data,
    domain,
    start,
    duration,
    Polygonal_rate_operator,
    Inlet_operator,
    defaults_module=None,
):
    """Apply Rainfall, Surface and Catchment inflows to an Anuga ``domain``.

    Inflow.make_file() server-side resolves each feature's ``properties.data``
    via FeatureDataMixin (TASK-820) to one of:

      * a list of ``{timestamp: ISO8601, value: number}`` dicts (timeseries case),
      * a float (constant case, from ``data_constant`` or numeric legacy value),
      * ``None`` (no resolved value, e.g. a legacy free-text row with neither
        ``data_constant`` nor ``data_timeseries_id`` set).

    This function guards at the consumption boundary, mirroring the contract
    enforced by ``build_time_boundary_function``. Side effects:

      * For each Rainfall inflow, register a ``Polygonal_rate_operator`` on
        ``domain`` with a positive ``RAINFALL_FACTOR``.
      * For each Catchment polygon, register a ``Polygonal_rate_operator``
        with a *negative* ``RAINFALL_FACTOR`` (the catchment absorbs the rain
        falling on it and re-introduces it as a Surface inflow elsewhere).
      * For each Surface inflow line that is fully inside the boundary,
        register an ``Inlet_operator``.

    Raises:
        NotImplementedError: if a catchment is paired with a timeseries
            rainfall (catchments need a single uniform rate), or if more than
            one rainfall polygon is paired with a catchment.

    Returns a dict ``{feature_id: callable}`` of the inflow callables created,
    primarily for test introspection.
    """
    pd = import_optional("pandas")
    if defaults_module is None:
        defaults_module = defaults

    rainfall_inflow_polygons = (input_data.get('rainfall') or {}).get('features', []) or []
    surface_inflow_lines = (input_data.get('inflow') or {}).get('features', []) or []
    catchment_polygons = (
        [feature for feature in input_data.get('catchment').get('features')]
        if input_data.get('catchment') else []
    )
    boundary_polygon = input_data.get('boundary_polygon')

    datetime_range = pd.date_range(start=start, periods=duration + 1, freq='s')
    inflow_dataframe = pd.DataFrame(datetime_range, columns=['timestamp'])
    inflow_functions = dict()

    def create_inflow_function(dataframe, name):
        def rain(time_in_seconds):
            t_sec = int(math.floor(time_in_seconds))
            return dataframe[name][t_sec]
        rain.__name__ = name
        return rain

    def _merge_timeseries(name, rows):
        """Merge a timeseries list of ``{timestamp, value}`` dicts into
        ``inflow_dataframe`` under column ``name``, ffill-aligned to the
        model's per-second timestamp index.
        """
        nonlocal inflow_dataframe
        new_dataframe = pd.DataFrame(rows)
        new_dataframe['timestamp'] = pd.to_datetime(new_dataframe['timestamp'])
        new_dataframe[name] = pd.to_numeric(new_dataframe['value'])
        if inflow_dataframe['timestamp'].dt.tz is None:
            inflow_dataframe['timestamp'] = inflow_dataframe['timestamp'].dt.tz_localize('UTC')
        if new_dataframe['timestamp'].dt.tz is None:
            new_dataframe['timestamp'] = new_dataframe['timestamp'].dt.tz_localize('UTC')
        inflow_dataframe = pd.merge(inflow_dataframe, new_dataframe, how='left', on='timestamp')
        inflow_dataframe.ffill(inplace=True)

    for inflow_polygon in rainfall_inflow_polygons:
        polygon_name = inflow_polygon.get('id')
        data = inflow_polygon.get('properties').get('data')
        if data is None:
            logger.warning(
                "Rainfall inflow %s has no resolved data (data_constant "
                "and data_timeseries_id both unset); skipping",
                polygon_name,
            )
            continue
        if isinstance(data, list):
            _merge_timeseries(polygon_name, data)
        else:
            inflow_dataframe[polygon_name] = float(data)
        inflow_function = create_inflow_function(inflow_dataframe, polygon_name)
        inflow_functions[polygon_name] = inflow_function
        geometry = _extract_polygon_outer_ring(inflow_polygon.get('geometry'))
        Polygonal_rate_operator(
            domain,
            rate=inflow_function,
            factor=defaults_module.RAINFALL_FACTOR,
            polygon=geometry,
            default_rate=0.00,
        )

    if len(rainfall_inflow_polygons) > 1 and len(catchment_polygons) > 0:
        # Fail-fast BEFORE the catchment loop so no Polygonal_rate_operator
        # side effects are registered on a configuration we cannot honour.
        raise NotImplementedError(
            'Cannot handle multiple rainfall polygons together with catchment '
            'hydrology.'
        )

    if len(rainfall_inflow_polygons) >= 1 and len(catchment_polygons) > 0:
        first_rainfall_data = rainfall_inflow_polygons[0].get('properties').get('data')
        if first_rainfall_data is None:
            raise NotImplementedError(
                'Catchment hydrology requires the first rainfall inflow to '
                'have a resolved data value, got None. Set a data_constant '
                'or data_timeseries_id on the first rainfall, or remove the '
                'catchment.'
            )
        if isinstance(first_rainfall_data, list):
            # Catchments use a SINGLE uniform rate, which is undefined for a
            # time-varying input. Surface the limitation clearly.
            raise NotImplementedError(
                'Catchment hydrology with a timeseries rainfall inflow is '
                'not supported. Use a constant rainfall value or remove the '
                'catchment.'
            )
        for catchment_polygon in catchment_polygons:
            uniform_rainfall_rate = float(first_rainfall_data)
            polygon_name = catchment_polygon.get('id')
            inflow_dataframe[polygon_name] = uniform_rainfall_rate
            inflow_function = create_inflow_function(inflow_dataframe, polygon_name)
            geometry = _extract_polygon_outer_ring(catchment_polygon.get('geometry'))
            # The catchment needs to be wholly in the domain:
            if check_coordinates_are_in_polygon(geometry, boundary_polygon):
                Polygonal_rate_operator(
                    domain,
                    rate=inflow_function,
                    factor=-defaults_module.RAINFALL_FACTOR,
                    polygon=geometry,
                    default_rate=0.00,
                )

    for inflow_line in surface_inflow_lines:
        polyline_name = inflow_line.get('id')
        data = inflow_line.get('properties').get('data')
        if data is None:
            logger.warning(
                "Surface inflow %s has no resolved data (data_constant "
                "and data_timeseries_id both unset); skipping",
                polyline_name,
            )
            continue
        if isinstance(data, list):
            _merge_timeseries(polyline_name, data)
        else:
            inflow_dataframe[polyline_name] = float(data)
        inflow_function = create_inflow_function(inflow_dataframe, polyline_name)
        inflow_functions[polyline_name] = inflow_function
        geometry = _flatten_line_coordinates(inflow_line.get('geometry'))
        if check_coordinates_are_in_polygon(geometry, boundary_polygon):
            Inlet_operator(domain, geometry, Q=inflow_function)

    return inflow_functions


def post_process_sww(package_dir, run_args=None, output_raster_resolution=None):
    # TASK-1143: ANUGA result rasters (depth/velocity/depthIntegratedVelocity/stage)
    # intentionally keep NaN as their nodata value, NOT -9999.  NaN is propagated
    # by Make_Geotif (plot_utils.py nodata=numpy.nan) and relied upon by make_video
    # (np.ma.masked_invalid) and make_raster_diff (--NoDataValue=nan).  Switching to
    # -9999 would break both without any benefit — result layers are single-coverage
    # per run so there is no overlapping-edge problem.  The assertion block below
    # guards against a future anuga_core default flip.
    anuga = import_optional("anuga")
    util = anuga.utilities.plot_utils
    output_quantities = ['depth', 'velocity', 'depthIntegratedVelocity', 'stage']
    input_data = setup_input_data(package_dir)
    logger.critical(f'Generating output rasters on {anuga.myid}...')
    resolutions = list()
    if input_data.get('mesh_region'):
        for feature in input_data.get('mesh_region').get('features') or list():
            # logger.critical(f'{feature=}')
            resolutions.append(feature.get('properties').get('resolution'))
    logger.critical(f'{resolutions=}')
    if len(resolutions) == 0:
        resolutions = [input_data.get('resolution') or 1000]
    finest_grid_resolution = min(resolutions)
    logger.critical(f'raster output resolution: {finest_grid_resolution}m')

    epsg_integer = int(input_data['scenario_config'].get("epsg").split(":")[1]
                       if ":" in input_data['scenario_config'].get("epsg")
                       else input_data['scenario_config'].get("epsg"))
    interior_holes, _ = make_interior_holes_and_tags(input_data)
    util.Make_Geotif(
        swwFile=f"{input_data['output_directory']}/{input_data['run_label']}.sww",
        output_quantities=output_quantities,
        myTimeStep='all',
        CellSize=finest_grid_resolution,
        lower_left=None,
        upper_right=None,
        EPSG_CODE=epsg_integer,
        proj4string=None,
        velocity_extrapolation=True,
        min_allowed_height=defaults.MIN_ALLOWED_HEIGHT_M,
        output_dir=input_data['output_directory'],
        bounding_polygon=input_data['boundary_polygon'],
        internal_holes=interior_holes,
        verbose=False,
        k_nearest_neighbours=defaults.K_NEAREST_NEIGHBOURS,
        creation_options=[]
    )
    util.Make_Geotif(
        swwFile=f"{input_data['output_directory']}/{input_data['run_label']}.sww",
        output_quantities=output_quantities,
        myTimeStep='max',
        CellSize=finest_grid_resolution,
        lower_left=None,
        upper_right=None,
        EPSG_CODE=epsg_integer,
        proj4string=None,
        velocity_extrapolation=True,
        min_allowed_height=defaults.MIN_ALLOWED_HEIGHT_M,
        output_dir=input_data['output_directory'],
        bounding_polygon=input_data['boundary_polygon'],
        internal_holes=interior_holes,
        verbose=False,
        k_nearest_neighbours=defaults.K_NEAREST_NEIGHBOURS,
        creation_options=[]
    )

    # TASK-1143: guard against a future anuga_core default flip away from NaN.
    # Re-open the *_max.tif outputs and assert band 1 nodata is NaN.
    import math
    _rasterio = import_optional("rasterio")
    for _quantity in output_quantities:
        _max_tif = os.path.join(
            input_data['output_directory'],
            f"{input_data['run_label']}_{_quantity}_max.tif",
        )
        if os.path.isfile(_max_tif):
            with _rasterio.open(_max_tif) as _ds:
                _nodata = _ds.nodata
            assert _nodata is not None and math.isnan(_nodata), (
                f"Result raster '{_max_tif}' nodata is {_nodata!r}; expected NaN. "
                "ANUGA result rasters intentionally use NaN nodata — do not switch to -9999."
            )

    video_dir = f"{input_data['output_directory']}/videos/"
    if os.path.isdir(video_dir):
        shutil.rmtree(video_dir)
    logger.critical('Successfully generated depth, velocity, momentum outputs')


def make_video(input_directory_1, result_type):
    np = import_optional("numpy")
    rasterio = import_optional("rasterio")
    cv2 = import_optional("cv2")
    plt = import_optional("matplotlib.pyplot")
    pe = import_optional("matplotlib.patheffects")
    run_label_1 = os.path.basename(input_directory_1).replace('outputs', 'run')
    tif_files = glob.glob(f"{input_directory_1}/{run_label_1}_{result_type}_*.tif")
    tif_files = [tif_file for tif_file in tif_files if "_max" not in tif_file]
    tif_files.sort(key=lambda f: int(os.path.splitext(f)[0][-6:]))

    max_file = f"{input_directory_1}/{run_label_1}_{result_type}_max.tif"
    global_min = 0
    raster_data = rasterio.open(max_file).read(1)
    masked_data_raster_data = np.ma.masked_invalid(raster_data)
    global_max = masked_data_raster_data.max()
    image_files = list()
    image_directory = f"{input_directory_1}/videos"
    if not os.path.exists(image_directory):
        os.makedirs(image_directory, exist_ok=True)
    for i, file in enumerate(tif_files):
        raster = rasterio.open(file)
        data = raster.read(1)
        label = str(file).split('/')[-1]
        plt.figure(figsize=(10, 10))
        plt.imshow(data, cmap='viridis', vmin=global_min, vmax=global_max)
        plt.axis('off')
        plt.text(0, 0, label, color='white', fontsize=10, ha='left', va='top', path_effects=[pe.withStroke(linewidth=3, foreground='black')])
        img_file = f"{image_directory}/frame_{result_type}_{i:03d}.png"
        plt.savefig(img_file, dpi=300)
        image_files.append(img_file)
        plt.close()

    img = cv2.imread(image_files[0])
    height, width, _ = img.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f"{input_directory_1}/{run_label_1}_{result_type}.mp4", fourcc, 20.0, (width, height))

    for img_file in image_files:
        img = cv2.imread(img_file)
        out.write(img)
    out.release()


def make_comparison_video(input_directory_1, input_directory_2, result_type):
    np = import_optional("numpy")
    rasterio = import_optional("rasterio")
    cv2 = import_optional("cv2")
    plt = import_optional("matplotlib.pyplot")
    pe = import_optional("matplotlib.patheffects")
    run_label_1 = os.path.basename(input_directory_1).replace('outputs', 'run')
    run_label_2 = os.path.basename(input_directory_2).replace('outputs', 'run')

    tif_files_1 = glob.glob(f"{input_directory_1}/{run_label_1}_{result_type}_*.tif")
    tif_files_1 = sorted([tif_file for tif_file in tif_files_1 if "_max" not in tif_file],
                         key=lambda f: int(os.path.splitext(f)[0][-6:]))

    tif_files_2 = glob.glob(f"{input_directory_2}/{run_label_2}_{result_type}_*.tif")
    tif_files_2 = sorted([tif_file for tif_file in tif_files_2 if "_max" not in tif_file],
                         key=lambda f: int(os.path.splitext(f)[0][-6:]))

    assert len(tif_files_1) == len(tif_files_2), "Number of tif files in the two directories are not the same."

    max_file = f"{input_directory_1}/{run_label_1}_{result_type}_max.tif"
    global_min = 0
    raster_data = rasterio.open(max_file).read(1)
    masked_data_raster_data = np.ma.masked_invalid(raster_data)
    global_max = masked_data_raster_data.max()

    image_files_1 = list()
    image_files_2 = list()
    image_files_diff = list()

    image_directory_1 = f"{input_directory_1}/videos"
    image_directory_2 = f"{input_directory_2}/videos"
    image_directory_diff = f"{input_directory_1}/diff_videos"

    for directory in [image_directory_1, image_directory_2, image_directory_diff]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    for i, (file_1, file_2) in enumerate(zip(tif_files_1, tif_files_2)):
        raster_1 = rasterio.open(file_1).read(1)
        raster_2 = rasterio.open(file_2).read(1)

        diff = np.abs(raster_2 - raster_1)

        for file, image_directory, image_files, data in [(file_1, image_directory_1, image_files_1, raster_1),
                                                         (file_2, image_directory_2, image_files_2, raster_2),
                                                         (None, image_directory_diff, image_files_diff, diff)]:
            plt.figure(figsize=(10, 10))
            if "diff" in image_directory:
                cmap = "RdBu"
            else:
                cmap = "viridis"
            plt.imshow(data, cmap=cmap, vmin=global_min, vmax=global_max)
            plt.axis('off')
            if file is not None:
                label = str(file).split('/')[-1]
                plt.text(0, 0, label, color='white', fontsize=10, ha='left', va='top',
                         path_effects=[pe.withStroke(linewidth=3, foreground='black')])
            img_file = f"{image_directory}/frame_{result_type}_{i:03d}.png"
            plt.savefig(img_file, dpi=100)
            image_files.append(img_file)
            plt.close()

    img1 = cv2.imread(image_files_1[0])
    img2 = cv2.imread(image_files_2[0])
    img_diff = cv2.imread(image_files_diff[0])

    largest_width = max([img1.shape[1], img2.shape[1], img_diff.shape[1]])
    height, _, _ = img1.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f"{input_directory_1}/{run_label_1}_{result_type}_difference.avi", fourcc, 20.0,
                          (3 * largest_width, height))

    for file1, file2, file_diff in zip(image_files_1, image_files_2, image_files_diff):
        img1 = cv2.imread(file1)
        img2 = cv2.imread(file2)
        img_diff = cv2.imread(file_diff)

        def pad_image(img):
            padding = ((0, 0), (0, largest_width - img.shape[1]), (0, 0))
            return np.pad(img, padding, mode='constant')

        img1 = pad_image(img1)
        img2 = pad_image(img2)
        img_diff = pad_image(img_diff)

        combined_img = np.concatenate((img1, img_diff, img2), axis=1)
        out.write(combined_img)

    out.release()


class _V2LogHandler(logging.Handler):
    """Logging handler that ships records to the V2 ``/log/`` control endpoint.

    Replaces the legacy ``logging.handlers.HTTPHandler`` (V1 BasicAuth to
    ``/anuga/api/{p}/{s}/run/{r}/log/``) installed by ``setup_logger`` before
    TASK-989. Auth is the site-wide ``X-Internal-Token`` shared secret (RAW,
    no ``Bearer`` prefix — see ``gn_anuga.permissions.IsInternalComputeCaller``),
    carried on a single owned ``requests.Session`` reused across every emit.

    Mirrors ``HydrataCallback`` (callbacks.py): the V2 log endpoint accepts
    ``{message, levelname, created}`` and writes the formatted line to
    ``Run.log``; ``project_id``/``scenario_id`` are inferred server-side from
    the run row, so only ``run_id`` appears in the URL.

    Emit failures never propagate (logging must not break the run loop): a
    transport error is routed through ``logging.Handler.handleError``. The
    owned Session pool is released by ``close()`` (idempotent); ``setup_logger``
    pairs ``addHandler`` with ``removeHandler`` + ``close()`` on re-entry.
    """

    def __init__(self, control_server: str, run_id, token: str):
        super().__init__()
        from run_anuga._http import make_internal_session

        base = str(control_server).rstrip('/')
        self._log_url = f"{base}/api/v2/anuga/runs/{run_id}/log/"
        self._session = make_internal_session(token)

    def emit(self, record: logging.LogRecord) -> None:
        from run_anuga._http import post_to_control_server

        try:
            post_to_control_server(
                self._log_url,
                method='POST',
                data={
                    'message': self.format(record),
                    'levelname': record.levelname,
                    'created': record.created,
                },
                session=self._session,
                timeout=30,
            )
        except Exception:
            # Never let a log shipment break the run — route to the standard
            # logging error hook (respects logging.raiseExceptions, which is
            # False in production).
            self.handleError(record)

    def close(self) -> None:
        """Release the owned ``requests.Session`` and unregister the handler.

        Idempotent: a second call is a no-op (``self._session`` may already be
        closed; ``requests.Session.close()`` tolerates repeat calls).
        """
        try:
            self._session.close()
        except Exception:  # pragma: no cover — defensive
            pass
        super().close()


def setup_logger(input_data, username=None, password=None, batch_number=1):
    if not username or not password:
        username = os.environ.get('COMPUTE_USERNAME')
        password = os.environ.get('COMPUTE_PASSWORD')
    # Create handlers
    file_handler = logging.FileHandler(os.path.join(input_data['output_directory'], f'run_anuga_{batch_number}.log'))
    file_handler.setLevel(logging.DEBUG)

    # Avoid duplicate handlers when run_sim() is called multiple times.
    # Close removed network handlers so their owned Session connection pool
    # is released (the V2 handler below owns a requests.Session); FileHandler
    # also benefits from close() flushing/closing its file descriptor.
    for h in logger.handlers[:]:
        if isinstance(h, (logging.FileHandler, logging.handlers.HTTPHandler, _V2LogHandler)):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:  # pragma: no cover — defensive, never break setup
                pass

    # Add handlers to the logger
    logger.addHandler(file_handler)

    # Ship log lines to the control server over the V2 log endpoint using the
    # site-wide X-Internal-Token shared secret (NOT the legacy V1 BasicAuth
    # HTTPHandler, which 401'd against allauth on localhost and targeted the
    # /anuga/api/.../log/ URL that TASK-1184 will delete). The token is read
    # from the environment, matching how run.py builds HydrataCallback; the
    # username/password parameters are retained for signature back-compat but
    # are no longer used for the web log channel. When the token is absent
    # (e.g. a standalone CLI run) no web handler is installed and logging
    # stays file/console-only.
    token = os.environ.get('HYDRATA_INTERNAL_COMPUTE_TOKEN')
    control_server = input_data['scenario_config'].get('control_server')
    run_id = input_data['scenario_config'].get('run_id')
    if token and control_server and run_id:
        web_handler = _V2LogHandler(
            control_server=control_server,
            run_id=run_id,
            token=token,
        )
        web_handler.setLevel(logging.DEBUG)
        logger.addHandler(web_handler)
    logger.setLevel(logging.DEBUG)
    return logger


def burn_structures_into_raster(structures_filename, raster_filename, backup=True):
    if backup:
        shutil.copyfile(raster_filename, f"{raster_filename[:-4]}_original.tif")
    output = subprocess.run(["gdal_rasterize", "-burn", str(defaults.BUILDING_BURN_HEIGHT_M), "-add", structures_filename, raster_filename], capture_output=True, universal_newlines=True)
    print(output)
    if output.returncode != 0:
        raise RuntimeError(output.stderr)
    return True


def make_shp_from_polygon(boundary_polygon, epsg_code, shapefilepath, buffer=0):
    ogr = import_optional("osgeo.ogr")
    osr = import_optional("osgeo.osr")
    boundary_ring_geom = ogr.Geometry(ogr.wkbLinearRing)
    for point in boundary_polygon:
        boundary_ring_geom.AddPoint(point[0], point[1])
    boundary_ring_geom.AddPoint(boundary_polygon[0][0], boundary_polygon[0][1])
    boundary_polygon_geom = ogr.Geometry(ogr.wkbPolygon)
    boundary_polygon_geom.AddGeometry(boundary_ring_geom)
    logger.critical(f"{boundary_polygon_geom=}")

    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(shapefilepath)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)
    layer = ds.CreateLayer("mesh_region", srs, ogr.wkbPolygon)
    id_field = ogr.FieldDefn("id", ogr.OFTInteger)
    layer.CreateField(id_field)
    feature_defn = layer.GetLayerDefn()
    feature = ogr.Feature(feature_defn)
    feature.SetGeometry(boundary_polygon_geom)
    feature.SetField("id", 1)
    layer.CreateFeature(feature)
    feature = None
    ds = None


def snap_links_to_nodes(package_dir):
    print('snap_links_to_nodes')
    raise NotImplementedError


def calculate_hydrology(package_dir):
    shapely_geometry = import_optional("shapely.geometry")
    Point, Polygon = shapely_geometry.Point, shapely_geometry.Polygon
    prep = import_optional("shapely.prepared").prep
    input_data = setup_input_data(package_dir)

    # prepare Nodes
    node_points = list()
    for node in input_data.get('nodes').get('features'):
        node_point = Point(node.get('geometry').get('coordinates'))
        node_points.append(node_point)

    # assign catchments to nodes
    for index, catchment in enumerate(input_data.get('catchment').get('features')):
        catchment_polygon = prep(Polygon(_extract_polygon_outer_ring(catchment.get('geometry'))))
        node_candidate = list(filter(catchment_polygon.contains, node_points))
        if len(node_candidate) == 0:
            logger.critical(f"Catchment {catchment.get('id')} has no internal node")
            continue
        if len(node_candidate) > 1:
            raise IndexError(f"Catchment {catchment.get('id')} has more than one node")
        print('find', node_candidate)
        node_index = node_points.index(node_candidate[0])
        input_data['catchment']['features'][index]['node_id'] = input_data.get('nodes').get('features')[node_index].get('id')

    rainfall_features = (input_data.get('rainfall') or {}).get('features', []) or []
    if not rainfall_features:
        logger.warning(
            "calculate_hydrology: no rainfall features supplied; "
            "skipping surface_flow_m3_s assignment for catchments"
        )
        return True
    rainfall_steady_state_intensity_mm_hr = float(rainfall_features[0].get('properties').get('data'))
    rainfall_steady_state_intensity_m_s = rainfall_steady_state_intensity_mm_hr * 0.001 / 3600
    for index, catchment in enumerate(input_data.get('catchment').get('features')):
        area_m2 = Polygon(_extract_polygon_outer_ring(catchment.get('geometry'))).area
        input_data['catchment']['features'][index]['surface_flow_m3_s'] = 1.0 * rainfall_steady_state_intensity_m_s * area_m2

    # create surface inflows at catchment nodes
    for index, catchment in enumerate(input_data.get('catchment').get('features')):
        if catchment.get('surface_flow_m3_s'):
            node_id = catchment.get('node_id')
            location = None
            for node in input_data.get('nodes').get('features'):
                if node.get('id') == node_id:
                    location = node.get('geometry').get('coordinates')
            # create an inflow object
            start_point = [location[0] - 10, location[1]]
            end_point = [location[0] + 10, location[1]]
            coordinates = [start_point, end_point]
            inflow_object = make_new_inflow(node_id, coordinates, catchment.get('surface_flow_m3_s'))
            add_inflow_to_file(inflow_object, input_data.get('inflow_filename'))

    return True


def make_new_inflow(inflow_id, coordinates, flow):
    return {
        'type': 'Feature',
        'id': f'inf_16_inflow_01.{inflow_id}',
        'geometry': {
            'type': 'LineString',
            'coordinates': coordinates
        },
        'geometry_name': 'the_geom',
        'properties': {
            'fid': 1,
            'type': 'Surface',
            'data': str(flow),
            'description': None
        }
    }


def add_inflow_to_file(inflow_object, filepath):
    with open(filepath) as f:
        file_contents = json.load(f)
    file_contents['features'].append(inflow_object)
    with open(filepath, 'w') as json_file:
        json.dump(file_contents, json_file)
    return True


def check_coordinates_are_in_polygon(coordinates, polygon):
    if not coordinates:
        return False
    shapely_geometry = import_optional("shapely.geometry")
    Point, Polygon = shapely_geometry.Point, shapely_geometry.Polygon
    shapely_polgyon = Polygon(polygon)
    if isinstance(coordinates[0], float):
        coordinates = [coordinates]
    for point in coordinates:
        shapely_point = Point(point)
        if not shapely_polgyon.contains(shapely_point):
            return False
    return True


def generate_stac(output_directory, run_label, output_quantities, initial_time_iso_string,
                  aws_access_key_id=None, aws_secret_access_key=None, s3_bucket_name=None):
    # Deterministic no-op when required scenario_config inputs are missing.
    # Without this guard, datetime.fromisoformat(None) and the for-loop over
    # output_quantities raise on every legacy run (no initial_time/output_quantities
    # in older packages). Log once so operators can spot the coverage gap.
    if not initial_time_iso_string or not output_quantities:
        logger.info(
            "STAC generation skipped: missing required inputs "
            "(run_label=%s, initial_time_iso_string=%r, output_quantities=%r)",
            run_label, initial_time_iso_string, output_quantities,
        )
        return
    _explicit_creds = aws_access_key_id is not None
    # Fall back to Django settings if params not provided
    if not _explicit_creds:
        if isinstance(settings, dict):
            return  # No Django settings and no explicit creds — silently skip
        aws_access_key_id = getattr(settings, 'AWS_ACCESS_KEY_ID', None)
        aws_secret_access_key = getattr(settings, 'AWS_SECRET_ACCESS_KEY', None)
        s3_bucket_name = getattr(settings, 'ANUGA_S3_STAC_BUCKET_NAME', None)
    if not aws_access_key_id or not aws_secret_access_key or not s3_bucket_name:
        if not _explicit_creds:
            return  # Django settings incomplete — silently skip (matches old behavior)
        raise ValueError("AWS credentials required: pass aws_access_key_id, aws_secret_access_key, and s3_bucket_name")

    boto3 = import_optional("boto3")
    rasterio = import_optional("rasterio")
    pystac = import_optional("pystac")
    Item, Asset, Collection, MediaType = pystac.Item, pystac.Asset, pystac.Collection, pystac.MediaType
    Extent, SpatialExtent, TemporalExtent = pystac.Extent, pystac.SpatialExtent, pystac.TemporalExtent
    CatalogType, Catalog = pystac.CatalogType, pystac.Catalog
    from pystac.stac_io import DefaultStacIO, StacIO

    class S3StacIO(DefaultStacIO):
        def __init__(self):
            self.s3 = boto3.resource(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
            super().__init__()

        def read_text(self, source, *args, **kwargs) -> str:
            parsed = urlparse(source)
            if parsed.scheme == "s3":
                bucket = parsed.netloc
                key = parsed.path[1:]
                obj = self.s3.Object(bucket, key)
                return obj.get()["Body"].read().decode("utf-8")
            else:
                return super().read_text(source, *args, **kwargs)

        def write_text(self, dest, txt, *args, **kwargs) -> None:
            parsed = urlparse(dest)
            if parsed.scheme == "s3":
                bucket = parsed.netloc
                key = parsed.path[1:]
                self.s3.Object(bucket, key).put(Body=txt, ContentEncoding="utf-8")
            else:
                super().write_text(dest, txt, *args, **kwargs)

    StacIO.set_default(S3StacIO)

    s3_catalog_uri = f"s3://{s3_bucket_name}/{run_label}"
    catalog = Catalog(id=run_label, description=f"{run_label} - {initial_time_iso_string}")
    min_left, min_bottom, max_right, max_top = None, None, None, None
    min_datetime, max_datetime = None, None
    initial_time = datetime.datetime.fromisoformat(initial_time_iso_string)

    for result_type in output_quantities:
        tif_files = glob.glob(f"{output_directory}/{run_label}_{result_type}_*.tif")
        tif_files = [tif_file for tif_file in tif_files if "_max" not in tif_file]
        tif_files.sort(key=lambda f: int(os.path.splitext(f)[0][-6:]))
        items = []
        collection = Collection(
            id=f"{run_label}_{result_type}",
            description="test",
            extent=Extent(
                spatial=SpatialExtent([[-180, -90, 180, 90]]),
                temporal=TemporalExtent([["2028-01-01T00:00:00Z", None]]),
            ),
        )

        for tif_file in tif_files:
            item_name = os.path.basename(tif_file).split('.')[0]
            s3_resource = boto3.resource(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
            s3_bucket = s3_resource.Bucket(s3_bucket_name)
            key = f"{run_label}/{result_type}/{os.path.basename(tif_file)}"
            with open(tif_file, 'rb') as data:
                s3_bucket.upload_fileobj(data, key)
            s3_tif_url = f"https://{s3_bucket_name}.s3.us-west-2.amazonaws.com/{key}"
            model_time_sec = int(tif_file[-10:-4])
            time_elapsed = initial_time + datetime.timedelta(seconds=model_time_sec)
            with rasterio.open(tif_file) as dataset:
                bbox = dataset.bounds
            item = Item(
                id=item_name,
                geometry={},
                bbox=[bbox.left, bbox.bottom, bbox.right, bbox.top],
                datetime=time_elapsed,
                properties={}
            )
            asset = Asset(
                href=s3_tif_url,
                media_type=MediaType.GEOTIFF
            )
            item.add_asset(key='data', asset=asset)
            items.append(item)
            collection.add_item(item)

            if min_left is None or bbox.left < min_left:
                min_left = bbox.left
            if min_bottom is None or bbox.bottom < min_bottom:
                min_bottom = bbox.bottom
            if max_right is None or bbox.right > max_right:
                max_right = bbox.right
            if max_top is None or bbox.top > max_top:
                max_top = bbox.top
            if min_datetime is None or time_elapsed < min_datetime:
                min_datetime = time_elapsed
            if max_datetime is None or time_elapsed > max_datetime:
                max_datetime = time_elapsed
        collection.extent = Extent(
            spatial=SpatialExtent([min_left, min_bottom, max_right, max_top]),
            temporal=TemporalExtent([[min_datetime, max_datetime]])
        )
        catalog.add_child(collection)

    catalog.extent = Extent(
        spatial=SpatialExtent([min_left, min_bottom, max_right, max_top]),
        temporal=TemporalExtent([[min_datetime, max_datetime]])
    )

    catalog.normalize_and_save(
        root_href=s3_catalog_uri,
        catalog_type=CatalogType.SELF_CONTAINED
    )
