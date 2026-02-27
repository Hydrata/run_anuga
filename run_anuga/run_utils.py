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
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from run_anuga._imports import import_optional
from run_anuga import defaults
from run_anuga.config import ScenarioConfig

logger = logging.getLogger(__name__)

try:
    from django.conf import settings
except ImportError:
    settings = dict()


@dataclass
class RunContext:
    """Typed replacement for the (package_dir, username, password) tuple."""
    package_dir: str
    username: Optional[str] = None
    password: Optional[str] = None


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
    input_data['scenario_config'] = json.load(open(os.path.join(package_dir, 'scenario.json')))
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
    input_data['boundary'] = json.load(open(input_data['boundary_filename']))

    data_types = [
        'friction',
        'inflow',
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
            input_data[data_type] = json.load(open(filepath))

    elevation_filepath = os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get('elevation')}")
    if input_data['scenario_config'].get('elevation') and os.path.isfile(elevation_filepath):
        input_data['elevation_filename'] = elevation_filepath

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
        input_data = setup_input_data(package_dir)
        data['project'] = input_data['scenario_config'].get('project')
        data['scenario'] = input_data['scenario_config'].get('id')
        run_id = input_data['scenario_config'].get('run_id')
        control_server = input_data['scenario_config'].get('control_server')
        client = requests.Session()
        client.auth = requests.auth.HTTPBasicAuth(username, password)
        # logger.critical(f"hydrata.com post:{data}")
        response = client.patch(
            f"{control_server}anuga/api/{data['project']}/{data['scenario']}/run/{run_id}/",
            data=data,
            files=files
        )
        status_code = response.status_code
        if status_code >= 400:
            logger.error(f"Error updating web interface. HTTP code: {status_code} - {response.text}")


def _clip_and_resample(src_path, dst_path, cutline_path, resolution):
    """Clip raster to cutline shapefile and resample to target resolution.

    Replaces: gdalwarp -cutline ... -crop_to_cutline -of GTiff -r cubic -tr res res
    """
    rasterio = import_optional("rasterio")
    gpd = import_optional("geopandas")
    from rasterio.warp import reproject, Resampling
    from rasterio.features import geometry_mask
    import numpy as np

    polygon = gpd.read_file(cutline_path)
    bounds = polygon.total_bounds  # [minx, miny, maxx, maxy]
    width = max(1, int(round((bounds[2] - bounds[0]) / resolution)))
    height = max(1, int(round((bounds[3] - bounds[1]) / resolution)))
    dst_transform = rasterio.transform.from_bounds(
        bounds[0], bounds[1], bounds[2], bounds[3], width, height
    )

    with rasterio.open(src_path) as src:
        dst_data = np.empty((height, width), dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=src.crs,
            resampling=Resampling.cubic,
        )
        nodata = src.nodata if src.nodata is not None else -9999
        # Mask pixels outside cutline
        mask = geometry_mask(polygon.geometry, out_shape=(height, width),
                             transform=dst_transform, invert=True)
        dst_data[~mask] = nodata

        profile = src.profile.copy()
        profile.update({
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'transform': dst_transform,
            'nodata': nodata,
        })

    with rasterio.open(dst_path, 'w', **profile) as dst:
        dst.write(dst_data, 1)


def create_mesher_mesh(input_data):
    rasterio = import_optional("rasterio")
    gpd = import_optional("geopandas")
    mesher_mesh_filepath = os.path.join(input_data['output_directory'], f"{input_data['elevation_filename'].split('/')[-1][:-4]}.mesh") or ""
    if os.path.isfile(mesher_mesh_filepath):
        with open(mesher_mesh_filepath, 'r') as mesh_file:
            mesh_dict = json.load(mesh_file)
        mesh_size = len(mesh_dict['mesh']['elem'])
        return mesher_mesh_filepath, mesh_size
    # logger = setup_logger(input_data)
    logger.info("create_mesh running")
    with rasterio.open(input_data['elevation_filename']) as src:
        elevation_raster_resolution = src.transform.a
    user_resolution = float(input_data.get('resolution'))
    if input_data.get('structure_filename') and \
            _has_reflective_structures(input_data['structure_filename']):
        burn_structures_into_raster(input_data['structure_filename'], input_data['elevation_filename'], backup=False)
    mesh_region_shp_files = None
    minimum_triangle_area = max((user_resolution ** 2) / 2, (elevation_raster_resolution ** 2) / 2)
    mesher_bin = os.environ.get('MESHER_EXE', defaults.DEFAULT_MESHER_EXE)
    if input_data.get('mesh_region_filename'):
        mesh_region_shp_files = list()
        for feature in input_data.get('mesh_region')['features']:
            resolution = feature.get('properties').get('resolution')
            mesh_region_name = feature.get('id')
            mesh_region_directory = Path(input_data.get('mesh_region_filename')).parent.joinpath(f"{mesh_region_name}")
            os.mkdir(mesh_region_directory)
            shp_boundary_filepath = os.path.join(mesh_region_directory, f"{mesh_region_name}.shp")
            tif_mesh_region_filepath = os.path.join(mesh_region_directory, f"{mesh_region_name}.tif")
            epsg_code = int(input_data.get('mesh_region').get('crs').get('properties').get('name').split(':')[-1])
            make_shp_from_polygon(feature.get('geometry').get('coordinates')[0], epsg_code, shp_boundary_filepath)
            logger.debug(shp_boundary_filepath)
            _clip_and_resample(
                input_data['elevation_filename'],
                tif_mesh_region_filepath,
                shp_boundary_filepath,
                resolution
            )
            logger.debug(f"Clipped and resampled to {tif_mesh_region_filepath}")
            mesh_region_shp_triangles = os.path.join(mesh_region_directory, mesh_region_name, f"{mesh_region_name}_USM.shp")
            mesh_region_shp_extent = os.path.join(mesh_region_directory, mesh_region_name, f"line_{mesh_region_name}.shp")
            mesh_region_shp_files.append({
                'mesh_region_shp_triangles': mesh_region_shp_triangles,
                'resolution': resolution,
                'mesh_region_shp_extent': mesh_region_shp_extent,
            })

            text_blob = f"""
mesher_path = '{mesher_bin}'
dem_filename = '{tif_mesh_region_filepath}'
errormetric = 'rmse'
max_tolerance = {resolution/10}
max_area = {(resolution ** 2) / 2}
min_area = {minimum_triangle_area}
user_output_dir = ''
nworkers = 2
nworkers_gdal = 2
write_vtu = False
simplify = True
simplify_buffer = -1
simplify_tol = 10
"""
            mesher_config_filepath = f"{mesh_region_directory}/mesher_config.py"
            with open(mesher_config_filepath, "w+") as mesher_config:
                mesher_config.write(text_blob)
            logger.debug(f"{mesher_config_filepath=}")
            with open(mesher_config_filepath, "r") as config_file:
                logger.debug(config_file.read())
            logger.debug(f"python {mesher_bin}.py {mesher_config_filepath}")
            mesher_out = subprocess.run([
                sys.executable,
                f'{mesher_bin}.py',
                mesher_config_filepath
            ],
                capture_output=True,
                universal_newlines=True
            )
            logger.debug(f"***{tif_mesh_region_filepath} mesher_out***")
            logger.debug(mesher_out.stdout)
            logger.debug(mesher_out.stderr)
            if mesher_out.returncode != 0:
                raise UserWarning(mesher_out.stderr)


        # mesh_regions_tif_mask_filename = f"{input_data['mesh_region_filename'][:-5]}.tif"
        # mesh_regions_tif_mask = subprocess.run([
        #     "gdal_rasterize",
        #     "-a", "resolution",
        #     "-te", str(xmin), str(ymin), str(xmax), str(ymax),
        #     "-tr", str(xres), str(yres),
        #     input_data['mesh_region_filename'],
        #     mesh_regions_tif_mask_filename
        # ],
        #     capture_output=True,
        #     universal_newlines=True
        # )
        # logger.critical(mesh_regions_tif_mask)
        # logger.critical(mesh_regions_tif_mask.stdout)
        # if mesh_regions_tif_mask.returncode != 0:
        #     logger.critical(mesh_regions_tif_mask.stderr)
        #     raise UserWarning(mesh_regions_tif_mask.stderr)
    # the lowest triangle area we can have is 5m2 or the grid resolution squared

    max_area = defaults.MAX_TRIANGLE_AREA
    mesher_config_filepath = f"{input_data['output_directory']}/mesher_config.py"
    logger.debug(f"{mesher_config_filepath=}")
    max_rmse_tolerance = input_data['scenario_config'].get('max_rmse_tolerance', 1)
    text_blob = f"""mesher_path = '{mesher_bin}'
dem_filename = '../inputs/{input_data["elevation_filename"].split("/")[-1]}'
errormetric = 'rmse'
max_tolerance = {max_rmse_tolerance}
max_area = {max_area}
min_area = {minimum_triangle_area}
user_output_dir = ''
nworkers = 2
nworkers_gdal = 2
write_vtu = False
simplify = True
simplify_buffer = -1
simplify_tol = 10
"""
#
#     if mesh_region_tif_files:
#         text_blob += f"""
# parameter_files = {{
#    'mesh_regions': {{
#        'file': '{mesh_region_tif_files[0][0]}',
#        'method': 'mean',
#        'tolerance': -1
#        }},
# }}
# """

    if mesh_region_shp_files:
        mesh_region_shp_files.sort(key=lambda mesh_region: mesh_region.get('resolution'), reverse=True)
        base_combined_filename = os.path.join(Path(input_data.get('mesh_region_filename')).parent, 'mesh_regions_combined')
        base_file_name = mesh_region_shp_files[0].get('mesh_region_shp_triangles')[:-4]
        new_combined_shp_name = None
        shutil.copy(f"{base_file_name}.shp", f"{base_combined_filename}_0.shp")
        shutil.copy(f"{base_file_name}.shx", f"{base_combined_filename}_0.shx")
        shutil.copy(f"{base_file_name}.prj", f"{base_combined_filename}_0.prj")
        shutil.copy(f"{base_file_name}.dbf", f"{base_combined_filename}_0.dbf")
        import pandas as pd
        for index, mesh_region in enumerate(mesh_region_shp_files):
            combined_layer_path = f"{base_combined_filename}_{index}.shp"
            combined = gpd.read_file(combined_layer_path)
            eraser = gpd.read_file(mesh_region.get('mesh_region_shp_extent'))
            print(70*'*')
            print(f"resolution: {mesh_region.get('resolution')}")
            print(f"combined_layer_path: {combined_layer_path}")
            print(f"eraser_layer: {mesh_region.get('mesh_region_shp_extent')}")
            print(f"combined_layer feature count: {len(combined)}")
            print(f"eraser_layer feature count: {len(eraser)}")

            # Erase eraser extent from combined triangles
            if len(combined) > 0 and len(eraser) > 0:
                erased = gpd.overlay(combined, eraser, how='difference')
            else:
                erased = combined

            print(f"after erase feature count: {len(erased)}")

            # Add new triangles from this mesh region
            new_triangles = gpd.read_file(mesh_region.get('mesh_region_shp_triangles'))
            print(f"new_triangles_layer: {mesh_region.get('mesh_region_shp_triangles')}")
            print(f"new_triangles feature count: {len(new_triangles)}")

            # Combine erased + new triangles (keep only geometry)
            final = gpd.GeoDataFrame(
                pd.concat([erased[['geometry']], new_triangles[['geometry']]], ignore_index=True),
                crs=combined.crs
            )
            new_combined_shp_name = f"{base_combined_filename}_{index + 1}.shp"
            final.to_file(new_combined_shp_name)

            print(f"final combined feature count: {len(final)}")
            print(70*'*')

        text_blob += f"""
constraints = {{
   'breaklines': {{
       'file': '{new_combined_shp_name}',
       }},
}}

"""
    with open(mesher_config_filepath, "w+") as mesher_config:
        mesher_config.write(text_blob)
    logger.debug(f"{mesher_config_filepath=}")
    with open(mesher_config_filepath, "r") as config_file:
        logger.debug(config_file.read())
    logger.debug(f"python {mesher_bin}.py {mesher_config_filepath}")
    try:
        mesher_out = subprocess.run([
            sys.executable,
            f'{mesher_bin}.py',
            mesher_config_filepath
        ],
            capture_output=True,
            universal_newlines=True
        )
        logger.debug("***mesher_out***")
        logger.debug(mesher_out.stdout)
        logger.debug(mesher_out.stderr)
        if mesher_out.returncode != 0:
            raise UserWarning(mesher_out.stderr)
    except ImportError:
        mesher_mesh_filepath = None
    logger.debug(f"{mesher_mesh_filepath=}")
    with open(mesher_mesh_filepath, 'r') as mesh_file:
        mesh_dict = json.load(mesh_file)
    mesh_size = len(mesh_dict['mesh']['elem'])
    return mesher_mesh_filepath, mesh_size


def create_anuga_mesh(input_data):
    anuga = import_optional("anuga")
    mesh_filepath = input_data['mesh_filepath']
    triangle_resolution = (input_data['scenario_config'].get('resolution') ** 2) / 2
    interior_regions = make_interior_regions(input_data)
    interior_holes, hole_tags = make_interior_holes_and_tags(input_data)
    bounding_polygon = input_data['boundary_polygon']
    boundary_tags = input_data['boundary_tags']
    logger.info("creating anuga_mesh")
    if input_data.get('structure_filename') and \
            _has_reflective_structures(input_data['structure_filename']):
        burn_structures_into_raster(input_data['structure_filename'], input_data['elevation_filename'], backup=False)
    # Do NOT pass mesh_geo_reference: ANUGA computes the offset from bounding polygon's
    # lower-left corner, keeping coordinates small. Passing Geo_reference(zone=N,xll=0,yll=0)
    # leaves absolute UTM values (~380000, ~6350000) which causes Triangle's float32
    # arithmetic to produce degenerate zero-area triangles near hole boundaries.
    anuga_mesh = anuga.pmesh.mesh_interface.create_mesh_from_regions(
        bounding_polygon=bounding_polygon,
        boundary_tags=boundary_tags,
        maximum_triangle_area=triangle_resolution,
        interior_regions=interior_regions,
        interior_holes=interior_holes,
        hole_tags=hole_tags,
        filename=mesh_filepath,
        use_cache=False,
        verbose=False,
        fail_if_polygons_outside=False
    )
    logger.info(f"{anuga_mesh.tri_mesh.triangles.size=}")
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
            mesh_polygon = mesh_region.get('geometry').get('coordinates')[0]
            mesh_resolution = mesh_region.get('properties').get('resolution')
            interior_regions.append((mesh_polygon, mesh_resolution,))
    return interior_regions


def make_interior_holes_and_tags(input_data):
    interior_holes = list()
    hole_tags = list()
    if input_data.get('structure'):
        for structure in input_data['structure']['features']:
            method = structure.get('properties', {}).get('method')
            if method == 'Holes':
                structure_polygon = structure['geometry']['coordinates'][0]
                interior_holes.append(structure_polygon)
                hole_tags.append({'reflective': list(range(len(structure_polygon)))})
            # Reflective → DEM-burned, not a mesh hole
            # Mannings → friction zone, not a mesh hole
    if not interior_holes:
        return None, None
    return interior_holes, hole_tags


def make_frictions(input_data):
    frictions = list()
    if input_data.get('structure'):
        for structure in input_data['structure']['features']:
            if structure.get('properties').get('method') == 'Mannings':
                structure_polygon = structure.get('geometry').get('coordinates')[0]
                frictions.append((structure_polygon, defaults.BUILDING_MANNINGS_N,))
    if input_data.get('friction'):
        for friction in input_data['friction']['features']:
            friction_polygon = friction.get('geometry').get('coordinates')[0]
            friction_value = friction.get('properties').get('mannings')
            frictions.append((friction_polygon, friction_value,))
    frictions.append(['All', defaults.DEFAULT_MANNINGS_N])
    return frictions


def compute_yieldstep(duration):
    """Calculate yield step interval for the simulation evolve loop.

    Returns an integer number of seconds, clamped to
    [MIN_YIELDSTEP_S, MAX_YIELDSTEP_S].
    """
    base_step = math.floor(duration / defaults.MAX_YIELDSTEPS)
    yieldstep = max(base_step, defaults.MIN_YIELDSTEP_S)
    return min(yieldstep, defaults.MAX_YIELDSTEP_S)


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


def create_boundary_polygon_from_boundaries(boundaries_geojson):
    from shapely.geometry import shape
    if boundaries_geojson.get('crs'):
        _epsg_code = boundaries_geojson.get('crs').get('properties').get('name').split(':')[-1]
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
        # Collect a list of the coordinates associated with each boundary tag:
        feature_coordinates = feature.get('geometry').get('coordinates')
        for coordinate in feature_coordinates:
            all_x_coordinates.append(coordinate[0])
            all_y_coordinates.append(coordinate[1])
    # Find the center of our project
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
        geometry = shape(feature.get('geometry'))
        centroid = [geometry.centroid.x, geometry.centroid.y]
        base = centroid[0] - mid_x
        height = centroid[1] - mid_y
        # the angle in polar coordinates will sort our boundary lines into the correct order
        angle = math.atan2(height, base)
        line_list.append({
            "centroid": centroid,
            "boundary": feature.get('properties').get('boundary'),
            "id": feature.get('id'),
            "angle": angle,
            "coordinates": feature.get('geometry').get('coordinates')
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


def post_process_sww(package_dir, run_args=None, output_raster_resolution=None):
    anuga = import_optional("anuga")
    util = anuga.utilities.plot_utils
    output_quantities = ['depth', 'velocity', 'depthIntegratedVelocity', 'stage']
    input_data = setup_input_data(package_dir)
    logger.info(f'Generating output rasters on {anuga.myid}...')
    resolutions = list()
    if input_data.get('mesh_region'):
        for feature in input_data.get('mesh_region').get('features') or list():
            # logger.critical(f'{feature=}')
            resolutions.append(feature.get('properties').get('resolution'))
    logger.debug(f'{resolutions=}')
    if len(resolutions) == 0:
        resolutions = [input_data.get('resolution') or 1000]
    finest_grid_resolution = min(resolutions)
    logger.info(f'raster output resolution: {finest_grid_resolution}m')

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
    _output_directory = input_data['output_directory']
    _run_label = input_data['run_label']
    _initial_time_iso_string = input_data['scenario_config'].get('model_start', "1970-01-01T00:00:00+00:00")
    # generate_stac(output_directory, run_label, output_quantities, initial_time_iso_string)
    # for result_type in output_quantities:
        # make_video(output_directory, run_label, result_type)

    video_dir = f"{input_data['output_directory']}/videos/"
    if os.path.isdir(video_dir):
        shutil.rmtree(video_dir)
    logger.info('Successfully generated depth, velocity, momentum outputs')


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


def setup_logger(input_data, username=None, password=None, batch_number=1):
    """Deprecated — use :func:`run_anuga.logging_setup.configure_simulation_logging` instead."""
    warnings.warn(
        "setup_logger() is deprecated. Use configure_simulation_logging() from run_anuga.logging_setup instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from run_anuga.logging_setup import configure_simulation_logging
    return configure_simulation_logging(input_data['output_directory'], batch_number)


def _has_reflective_structures(structures_filename):
    """Return True if any structure feature has method='Reflective'."""
    with open(structures_filename) as f:
        data = json.load(f)
    return any(
        feat.get('properties', {}).get('method') == 'Reflective'
        for feat in data.get('features', [])
    )


def burn_structures_into_raster(structures_filename, raster_filename, backup=True):
    """Burn Reflective structure geometries into a raster file (additive).

    Only features with method='Reflective' are burned; Mannings and Holes
    buildings are handled separately and must not alter the DEM.
    """
    rasterio = import_optional("rasterio")
    from rasterio.features import rasterize

    if backup:
        shutil.copyfile(raster_filename, f"{raster_filename[:-4]}_original.tif")

    with open(structures_filename) as f:
        structures = json.load(f)

    shapes = []
    for feature in structures.get('features', []):
        if feature.get('properties', {}).get('method') != 'Reflective':
            continue   # only burn Reflective buildings
        geom = feature.get('geometry')
        if geom:
            shapes.append((geom, defaults.BUILDING_BURN_HEIGHT_M))

    if not shapes:
        return True

    with rasterio.open(raster_filename, 'r+') as src:
        existing = src.read(1)
        burn = rasterize(
            shapes,
            out_shape=src.shape,
            transform=src.transform,
            fill=0,
            dtype=existing.dtype
        )
        src.write(existing + burn, 1)

    return True


def make_shp_from_polygon(boundary_polygon, epsg_code, shapefilepath, buffer=0):
    gpd = import_optional("geopandas")
    from shapely.geometry import Polygon

    poly = Polygon(boundary_polygon)
    if buffer > 0:
        poly = poly.buffer(buffer)
    logger.debug(f"make_shp_from_polygon: {poly}")

    gdf = gpd.GeoDataFrame({'id': [1]}, geometry=[poly], crs=f"EPSG:{epsg_code}")
    gdf.to_file(shapefilepath)


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
        catchment_polygon = prep(Polygon(catchment.get('geometry').get('coordinates')[0]))
        node_candidate = list(filter(catchment_polygon.contains, node_points))
        if len(node_candidate) == 0:
            logger.warning(f"Catchment {catchment.get('id')} has no internal node")
            continue
        if len(node_candidate) > 1:
            raise IndexError(f"Catchment {catchment.get('id')} has more than one node")
        print('find', node_candidate)
        node_index = node_points.index(node_candidate[0])
        input_data['catchment']['features'][index]['node_id'] = input_data.get('nodes').get('features')[node_index].get('id')

    # assign rainfall to catchments
    def rainfall_filter(inflow_feature):
        return inflow_feature.get('properties').get('type')

    rainfall_inflows = list(filter(rainfall_filter, input_data.get('inflow').get('features')))
    rainfall_steady_state_intensity_mm_hr = float(rainfall_inflows[0].get('properties').get('data'))
    rainfall_steady_state_intensity_m_s = rainfall_steady_state_intensity_mm_hr * 0.001 / 3600
    for index, catchment in enumerate(input_data.get('catchment').get('features')):
        area_m2 = Polygon(catchment.get('geometry').get('coordinates')[0]).area
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
    file_contents = json.load(open(filepath))
    file_contents['features'].append(inflow_object)
    with open(filepath, 'w') as json_file:
        json.dump(file_contents, json_file)
    return True


def check_coordinates_are_in_polygon(coordinates, polygon):
    shapely_geometry = import_optional("shapely.geometry")
    Point, Polygon = shapely_geometry.Point, shapely_geometry.Polygon
    shapely_polygon = Polygon(polygon)
    if isinstance(coordinates[0], float):
        coordinates = [coordinates]
    for point in coordinates:
        shapely_point = Point(point)
        if not shapely_polygon.contains(shapely_point):
            return False
    return True


def generate_stac(output_directory, run_label, output_quantities, initial_time_iso_string,
                  aws_access_key_id=None, aws_secret_access_key=None, s3_bucket_name=None):
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
