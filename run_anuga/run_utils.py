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

    boundary_polygon, boundary_tags = create_boundary_polygon_from_boundaries(input_data['boundaries'])
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
        response = client.patch(
            f"https://hydrata.com/anuga/api/{data['project']}/{data['scenario']}/run/{run_id}/",
            data=data,
            files=files
        )
        logger.info(response.status_code, response.content)


def create_boundary_polygon_from_boundaries(boundaries_geojson):
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

    # Find the center of our project, the sort the boundary lines in clockwise direction around it
    max_x = max(all_x_coordinates)
    max_y = max(all_y_coordinates)
    min_x = min(all_x_coordinates)
    min_y = min(all_y_coordinates)
    mid_x = max_x - (max_x - min_x) / 2
    mid_y = max_y - (max_y - min_y) / 2
    line_list = list()
    for index, feature in enumerate(boundaries_geojson.get('features')):
        if feature.get('properties').get('location') != "External":
            continue
        geometry = ogr.CreateGeometryFromJson(json.dumps(feature.get('geometry')))
        centroid = json.loads(geometry.Centroid().ExportToJson()).get('coordinates')
        line_list.append({
            "centroid": centroid,
            "boundary": feature.get('properties').get('boundary'),
            "id": feature.get('id'),
            "angle": math.atan((centroid[0] - mid_x)/(centroid[1] - mid_y)),
            "coordinates": feature.get('geometry').get('coordinates')
        })
    line_list.sort(key=lambda line: line.get('angle'))

    # Now join all our lines in clockwise order
    boundary_polygon = list()
    for index, line in enumerate(line_list):
        boundary_polygon.extend(line.get("coordinates"))
        boundary_tags[line.get("boundary")].append(index)

    # Save a GeoJson copy so we can debug/test what happened, if it's not working for any reason
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for coordinate in boundary_polygon:
        ring.AddPoint(coordinate[0], coordinate[1])
    geojson = ring.ExportToJson()
    filepath = os.path.join(os.getcwd(), 'test.geojson')
    print(filepath)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(json.loads(geojson), f, ensure_ascii=False, indent=4)

    return boundary_polygon, boundary_tags
