import shutil
import traceback

import anuga
import argparse
import json
import logging
import math
import subprocess
import os
import requests
import numpy as np

from pathlib import Path
from osgeo import ogr, gdal
from anuga.utilities import plot_utils as util

from celery.utils.log import get_task_logger
logger = get_task_logger('run')


def is_dir_check(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def setup_input_data(package_dir):
    if not os.path.isfile(os.path.join(package_dir, 'scenario.json')):
        raise FileNotFoundError(f'Could not find "scenario.json" in {package_dir}')

    input_data = dict()
    input_data['scenario_config'] = json.load(open(os.path.join(package_dir, 'scenario.json')))
    project_id = input_data['scenario_config'].get('project')
    scenario_id = input_data['scenario_config'].get('id')
    run_id = input_data['scenario_config'].get('run_id')
    input_data['run_label'] = f"run_{project_id}_{scenario_id}_{run_id}"
    input_data['output_directory'] = os.path.join(package_dir, f'outputs_{project_id}_{scenario_id}_{run_id}')
    input_data['mesh_filepath'] = f"{input_data['output_directory']}/run_{scenario_id}_{run_id}.msh"
    Path(input_data['output_directory']).mkdir(parents=True, exist_ok=True)

    input_data['boundary'] = json.load(
        open(os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get('boundary')}"))
    )
    if input_data['scenario_config'].get('friction'):
        input_data['friction'] = json.load(
            open(os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get('friction')}"))
        )
    if input_data['scenario_config'].get('inflow'):
        input_data['inflow'] = json.load(
            open(os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get('inflow')}"))
        )
    if input_data['scenario_config'].get('structure'):
        input_data['structure'] = json.load(
            open(os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get('structure')}"))
        )
    if input_data['scenario_config'].get('mesh_region'):
        input_data['mesh_region'] = json.load(
            open(os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get('mesh_region')}"))
        )
    if input_data['scenario_config'].get('elevation'):
        input_data['elevation_filename'] = os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get('elevation')}")

    if input_data['scenario_config'].get('maximum_triangle_area'):
        input_data['maximum_triangle_area'] = input_data['scenario_config'].get('maximum_triangle_area')

    boundary_polygon, boundary_tags = create_boundary_polygon_from_boundaries(
        input_data['boundary']
    )
    input_data['boundary_polygon'] = boundary_polygon
    input_data['boundary_tags'] = boundary_tags
    return input_data


def update_web_interface(run_args, data, files=None):
    package_dir, username, password = run_args
    if username and password:
        input_data = setup_input_data(package_dir)
        data['project'] = input_data['scenario_config'].get('project')
        data['scenario'] = input_data['scenario_config'].get('id')
        run_id = input_data['scenario_config'].get('run_id')
        client = requests.Session()
        client.auth = requests.auth.HTTPBasicAuth(username, password)
        # logger.info(f"hydrata.com post:{data}")
        response = client.patch(
            f"https://hydrata.com/anuga/api/{data['project']}/{data['scenario']}/run/{run_id}/",
            data=data,
            files=files
        )
        status_code = response.status_code
        logger.info(f"hydrata.com response: {status_code}")


def create_mesh(input_data):
    # logger = setup_logger(input_data)
    logger.info(f"create_mesh running")
    raster = gdal.Open(input_data['elevation_filename'])
    gt = raster.GetGeoTransform()
    grid_resolution = gt[1]
    # the lowest triangle area we can have is 5m2 or the grid resolution squared
    minimum_triangle_area = 5 if (grid_resolution ** 2) < 5 else (grid_resolution ** 2)
    mesh_filepath = input_data['mesh_filepath']
    # maximum_triangle_area = input_data.get('maximum_triangle_area') or 1000000
    maximum_triangle_area = 1000000
    interior_regions = make_interior_regions(input_data)
    interior_holes, hole_tags = make_interior_holes_and_tags(input_data)
    bounding_polygon = input_data['boundary_polygon']
    boundary_tags = input_data['boundary_tags']
    logger.info(f"creating anuga_mesh")
    anuga_mesh = anuga.pmesh.mesh_interface.create_mesh_from_regions(
        bounding_polygon=bounding_polygon,
        boundary_tags=boundary_tags,
        maximum_triangle_area=maximum_triangle_area,
        interior_regions=interior_regions,
        interior_holes=interior_holes,
        hole_tags=hole_tags,
        filename=mesh_filepath,
        use_cache=False,
        verbose=True,
        fail_if_polygons_outside=False
    )
    anuga_mesh_size = anuga_mesh.tri_mesh.triangles.size
    logger.info(f"{anuga_mesh_size=}")
    mesher_mesh_filepath = None
    mesher_bin = os.environ.get('MESHER_EXE')
    if mesher_bin:
        mesher_config_filepath = f"{input_data['output_directory']}/mesher_config.py"
        logger.info(f"{mesher_config_filepath=}")
        max_rmse_tolerance = input_data['scenario_config'].get('max_rmse_tolerance', 1)
        make_mesher_config_file(
            mesher_config_filepath,
            f"{input_data['output_directory']}/",
            input_data['elevation_filename'],
            mesher_bin,
            max_rmse_tolerance,
            minimum_triangle_area,
            maximum_triangle_area
        )
        logger.info(f"{mesher_config_filepath=}")
        logger.info("*" * 70)
        with open(mesher_config_filepath, "r") as config_file:
            logger.info(config_file.read())
        logger.info("*" * 70)
        logger.info(f"python {mesher_bin}.py {mesher_config_filepath}")
        try:
            logger.info("from mesher.mesher import main as mesher_main")
            from mesher.mesher import main as mesher_main
            import io
            from contextlib import redirect_stdout
            mesher_mesh_filepath = os.path.join(input_data['output_directory'], f"{input_data['elevation_filename'].split('/')[-1][:-4]}.mesh")
            logger.info(f"{mesher_mesh_filepath=}")
            # temp_stdout_obj = io.StringIO()
            # with redirect_stdout(temp_stdout_obj):
            #     mesher_main(mesher_config_filepath)
            # mesher_out = temp_stdout_obj.getvalue()
            # try:
            #     logger.info("mesher_main(mesher_config_filepath)")
            #     mesher_main(mesher_config_filepath)
            #     logger.info("success - mesher_main(mesher_config_filepath)")
            # except Exception as e:
            #     logger.info(traceback.format_exc())
            mesher_out = subprocess.run(['python', f'{mesher_bin}', mesher_config_filepath], capture_output=True)
            logger.info(f"{mesher_out=}")
            logger.info("-" * 70)
        except ImportError:
            mesher_mesh_filepath = None
    logger.info(f"{mesher_mesh_filepath=}")
    logger.info(f"{os.path.isfile(mesher_mesh_filepath)}")
    return anuga_mesh, mesher_mesh_filepath


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
            if structure.get('properties').get('method') == 'Mannings':
                continue
            structure_polygon = structure.get('geometry').get('coordinates')[0]
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
    frictions = list()
    if input_data.get('structure'):
        for structure in input_data['structure']['features']:
            if structure.get('properties').get('method') == 'Mannings':
                structure_polygon = structure.get('geometry').get('coordinates')[0]
                frictions.append((structure_polygon, 10,))  # TODO: maybe make building value customisable
    if input_data.get('friction'):
        for friction in input_data['friction']['features']:
            friction_polygon = friction.get('geometry').get('coordinates')[0]
            friction_value = friction.get('properties').get('mannings')
            frictions.append((friction_polygon, friction_value,))
    frictions.append(['All', 0.04])  # TODO: make default value customisable
    return frictions


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
    geometry_collection = ogr.Geometry(ogr.wkbGeometryCollection)
    epsg_code = boundaries_geojson.get('crs').get('properties').get('name').split(':')[-1]
    # Create a dict of the available boundary tags
    boundary_tags = dict()
    all_x_coordinates = list()
    all_y_coordinates = list()
    for index, feature in enumerate(boundaries_geojson.get('features')):
        if feature.get('properties').get('location') != "External":
            continue
        boundary_tags[feature.get('properties').get('boundary')] = []
        geometry = ogr.CreateGeometryFromJson(json.dumps(feature.get('geometry')))
        geometry_collection.AddGeometry(geometry)
        # Collect a list of the coordinates associated with each boundary tag:
        feature_coordinates = feature.get('geometry').get('coordinates')
        for coordinate in feature_coordinates:
            all_x_coordinates.append(coordinate[0])
            all_y_coordinates.append(coordinate[1])
    srs = ogr.osr.SpatialReference()
    epsg_integer = int(epsg_code.split(':')[1] if ':' in epsg_code else epsg_code)
    srs.ImportFromEPSG(epsg_integer)

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
        geometry = ogr.CreateGeometryFromJson(json.dumps(feature.get('geometry')))
        centroid = json.loads(geometry.Centroid().ExportToJson()).get('coordinates')
        base = centroid[0] - mid_x
        height = centroid[1] - mid_y
        # the angle in polar coordinates will sort our boundary lines into the correct order
        angle = math.atan(height/base) + correction_for_polar_quadrants(base, height)
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
    for line in line_list:
        for coordinate in line.get("coordinates"):
            boundary_polygon.append(coordinate)
            boundary_tags[line.get("boundary")].append(counter)
            boundary_tags_list.append(lookup_boundary_tag(counter, boundary_tags))
            counter += 1

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

    return boundary_polygon, boundary_tags


def post_process_sww(package_dir, run_args=None, output_raster_resolution=None):
    input_data = setup_input_data(package_dir)
    logger.info(f'Generating output rasters on {anuga.myid}...')
    raster = gdal.Open(input_data['elevation_filename'])
    gt = raster.GetGeoTransform()
    # resolution = 1 if math.floor(gt[1] / 4) == 0 else math.floor(gt[1] / 4)
    resolutions = list()
    if input_data.get('mesh_region'):
        for feature in input_data.get('mesh_region').get('features') or list():
            # logger.info(f'{feature=}')
            resolutions.append(feature.get('properties').get('resolution'))
    logger.info(f'{resolutions=}')
    if len(resolutions) == 0:
        resolutions = [input_data.get('maximum_triangle_area') or 1000]
    finest_grid_resolution = math.floor(math.sqrt(2 * min(resolutions)))
    logger.info(f'raster resolution: {finest_grid_resolution}m')
    epsg_integer = int(input_data['scenario_config'].get("epsg").split(":")[1]
                       if ":" in input_data['scenario_config'].get("epsg")
                       else input_data['scenario_config'].get("epsg"))
    interior_holes, _ = make_interior_holes_and_tags(input_data)
    util.Make_Geotif(
        swwFile=f"{input_data['output_directory']}/{input_data['run_label']}.sww",
        output_quantities=['depth', 'velocity', 'depthIntegratedVelocity'],
        myTimeStep='max',
        CellSize=finest_grid_resolution,
        lower_left=None,
        upper_right=None,
        EPSG_CODE=epsg_integer,
        proj4string=None,
        velocity_extrapolation=True,
        min_allowed_height=1.0e-05,
        output_dir=input_data['output_directory'],
        bounding_polygon=input_data['boundary_polygon'],
        internal_holes=interior_holes,
        verbose=False,
        k_nearest_neighbours=3,
        creation_options=[]
    )
    logger.info('Successfully generated depth, velocity, momentum outputs')
    if run_args:
        update_web_interface(
            run_args,
            data={"status": "uploading depth_max"},
            files={
                "tif_depth_max": open(
                    f"{input_data['output_directory']}/{input_data['run_label']}_depth_max.tif",
                    'rb'
                )
            }
        )
        update_web_interface(
            run_args,
            data={"status": "uploading depthIntegratedVelocity_max"},
            files={
                "tif_depth_integrated_velocity_max": open(
                    f"{input_data['output_directory']}/{input_data['run_label']}_depthIntegratedVelocity_max.tif",
                    'rb'
                )
            }
        )
        update_web_interface(
            run_args,
            data={"status": "uploading velocity_max"},
            files={
                "tif_velocity_max": open(
                    f"{input_data['output_directory']}/{input_data['run_label']}_velocity_max.tif",
                    'rb'
                )
            }
        )
        logger.info('Successfully uploaded outputs')


def zip_result_package(package_dir, username=None, password=None, remove=False):
    input_data = setup_input_data(package_dir)
    zip_filename = f"{input_data.get('scenario_config').get('run_id')}_{input_data.get('scenario_config').get('id')}_{input_data.get('scenario_config').get('project')}_results"
    zip_directory = Path(package_dir).parent.absolute()
    zip_filepath = str(Path(zip_directory, zip_filename))
    shutil.make_archive(zip_filepath, 'zip', package_dir)
    if username and password:
        run_args = (package_dir, username, password)
        update_web_interface(
            run_args,
            data={"status": "archiving results"},
            files={
                "result_package": open(
                    f"{zip_filepath}.zip",
                    'rb'
                )
            }
        )
    if remove:
        shutil.rmtree(package_dir)


def make_mesher_config_file(
    mesher_config_filepath,
    user_output_dir,
    dem_filepath,
    mesher_bin,
    max_rmse_tolerance,
    min_triangle_area,
    maximum_triangle_area
):
    text_blob = f"""mesher_path = '{mesher_bin}'
dem_filename = '../inputs/{dem_filepath.split("/")[-1]}'
errormetric = 'rmse'
max_tolerance = {max_rmse_tolerance}  # 1m max RMSE between triangle and underlying elevation
max_area = {maximum_triangle_area}  # Effectively unlimited upper area -- allow tolerance check to refine it further
min_area = {min_triangle_area}  # triangle area below which we will no longer refine, regardless of max_tolerance
user_output_dir = ''
nworkers = 2
nworkers_gdal = 2
write_vtu = False
simplify = True
simplify_tol = 10
"""
    with open(mesher_config_filepath, "w+") as mesher_config:
        mesher_config.write(text_blob)
    return True

a="""
--poly-file /opt/deploy/vagrant/include/package_testfour_testone_567/outputs_252_172_567//opt/deploy/vagrant/include/package_testfour_testone_567/outputs_252_172_567/ele_290_utm_ATestFourDem/PLGSele_290_utm_ATestFourDem.poly
--tolerance 1
--raster /opt/deploy/vagrant/include/package_testfour_testone_567/outputs_252_172_567//opt/deploy/vagrant/include/package_testfour_testone_567/outputs_252_172_567/ele_290_utm_ATestFourDem/ele_290_utm_ATestFourDem_projected.tif
--area 1000000
--min-area 5
--error-metric rmse
--lloyd 0
--interior-plgs-file /opt/deploy/vagrant/include/package_testfour_testone_567/outputs_252_172_567//opt/deploy/vagrant/include/package_testfour_testone_567/outputs_252_172_567/ele_290_utm_ATestFourDem/interior_PLGS.geojson
"""


def setup_logger(input_data, username=None, password=None):
    if not username and password:
        username = os.environ.get('COMPUTE_USERNAME')
        password = os.environ.get('COMPUTE_PASSWORD')
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(input_data['output_directory'], 'run_anuga.log'))
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    # # Create formatters and add it to handlers
    # console_format = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    # file_format = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    # console_handler.setFormatter(console_format)
    # file_handler.setFormatter(file_format)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    if username and password:
        web_handler = logging.handlers.HTTPHandler(
            host='hydrata.com',
            url=f"/anuga/api/{input_data['scenario_config'].get('project')}/{input_data['scenario_config'].get('id')}/run/{input_data['scenario_config'].get('run_id')}/log/",
            method='POST',
            secure=True,
            credentials=(username, password,)
        )
        web_handler.setLevel(logging.DEBUG)
        logger.addHandler(web_handler)
    return logger
