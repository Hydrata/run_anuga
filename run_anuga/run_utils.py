import argparse
import math
import json
import logging
import os
import requests

from pathlib import Path
from osgeo import ogr

logger = logging.getLogger(__name__)


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

    input_data['boundaries'] = json.load(
        open(os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get('boundary')}"))
    )
    input_data['friction_maps'] = json.load(
        open(os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get('friction')}"))
    )
    input_data['inflows'] = json.load(
        open(os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get('inflow')}"))
    )
    input_data['structures'] = json.load(
        open(os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get('structure')}"))
    )
    input_data['mesh_regions'] = json.load(
        open(os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get('mesh_region')}"))
    )
    input_data['elevation_filename'] = os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get('elevation')}")

    boundary_polygon, boundary_tags = create_boundary_polygon_from_boundaries(
        input_data['boundaries'],
        input_data['run_label'],
        input_data['scenario_config'].get('epsg'),
        package_dir
    )
    input_data['boundary_polygon'] = boundary_polygon
    input_data['boundary_tags'] = boundary_tags

    logger.debug(f"{input_data['boundaries']=}")
    logger.debug(f"{input_data['friction_maps']=}")
    logger.debug(f"{input_data['inflows']=}")
    logger.debug(f"{input_data['structures']=}")
    logger.debug(f"{input_data['mesh_regions']=}")
    logger.debug(f"{input_data['elevation_filename']=}")
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
        logger.info(f"hydrata.com post:{data}")
        response = client.patch(
            f"https://hydrata.com/anuga/api/{data['project']}/{data['scenario']}/run/{run_id}/",
            data=data,
            files=files
        )
        logger.info(f"hydrata.com response:{response.status_code}")


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


def create_boundary_polygon_from_boundaries(boundaries_geojson, run_label, epsg_code, package_dir):
    geometry_collection = ogr.Geometry(ogr.wkbGeometryCollection)

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
