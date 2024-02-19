import json
import time

import numpy

import anuga
import argparse
import math
import os
import pandas as pd
import traceback

from anuga import distribute, finalize, barrier, Inlet_operator
from anuga.utilities import quantity_setting_functions as qs
from anuga.operators.rate_operators import Polygonal_rate_operator
from .run_utils import is_dir_check, setup_input_data, update_web_interface, create_mesher_mesh, create_anuga_mesh, make_interior_holes_and_tags, \
    make_frictions, post_process_sww, zip_result_package, setup_logger, check_coordinates_are_in_polygon

from celery.utils.log import get_task_logger
logger = get_task_logger(__name__)


def run_sim(package_dir, username=None, password=None):
    run_args = package_dir, username, password
    input_data = setup_input_data(package_dir)
    output_stats = dict()
    logger = setup_logger(input_data, username, password)
    result_zip_path = None
    try:
        logger.info(f"{anuga.myid=}")
        if anuga.myid == 0:
            logger.info(f"update_web_interface - building mesh")
            update_web_interface(run_args, data={'status': 'building mesh'})
            domain = None
            if input_data['scenario_config'].get('simplify_mesh'):
                mesher_mesh_filepath, mesh_size = create_mesher_mesh(input_data)
                with open(mesher_mesh_filepath, 'r') as mesh_file:
                    mesh_dict = json.load(mesh_file)
                    vertex = mesh_dict['mesh']['vertex']
                    vertex = numpy.array(vertex)
                    elem = mesh_dict['mesh']['elem']
                    points = vertex[:, :2]
                    elev = vertex[:, 2]
                    domain = anuga.Domain(
                        points,
                        elem,
                        use_cache=False,
                        verbose=False
                    )
                    domain.set_quantity('elevation', elev, location='vertices')
                    update_web_interface(run_args, data={'mesh_triangle_count': mesh_size})
            else:
                anuga_mesh_filepath, mesh_size = create_anuga_mesh(input_data)
                domain = anuga.Domain(
                    mesh_filename=input_data['mesh_filepath'],
                    use_cache=False,
                    verbose=False,
                )
                poly_fun_pairs = [['Extent', input_data['elevation_filename']]]
                elevation_function = qs.composite_quantity_setting_function(
                    poly_fun_pairs,
                    domain,
                    nan_treatment='exception',
                )
                domain.set_quantity('elevation', elevation_function, verbose=False, alpha=0.99, location='centroids')
            if input_data['scenario_config'].get('store_mesh'):
                if getattr(domain, "dump_shapefile", None):
                    shapefile_name = f"{input_data['output_directory']}/{input_data['scenario_config'].get('run_id')}_{input_data['scenario_config'].get('id')}_{input_data['scenario_config'].get('project')}_mesh"
                    logger.info(f"mesh shapefile: {shapefile_name}")
                    domain.dump_shapefile(
                        shapefile_name=shapefile_name,
                        epsg_code=input_data['scenario_config'].get('epsg')
                    )
            domain.set_name(input_data['run_label'])
            domain.set_datadir(input_data['output_directory'])
            domain.set_minimum_storable_height(0.005)
            frictions = make_frictions(input_data)
            friction_function = qs.composite_quantity_setting_function(
                frictions,
                domain
            )
            domain.set_quantity('friction', friction_function, verbose=False)
            domain.set_quantity('stage', 0.0, verbose=False)

            update_web_interface(run_args, data={'status': 'created mesh'})
        else:
            domain = None
        # logger.info(f"domain on anuga.myid {anuga.myid}: {domain}")
        barrier()
        domain = distribute(domain, verbose=False)
        # logger.info(f"domain on anuga.myid {anuga.myid} after distribute(): {domain}")
        default_boundary_maps = {
            'exterior': anuga.Dirichlet_boundary([0, 0, 0]),
            'interior': anuga.Reflective_boundary(domain),
            'Dirichlet': anuga.Dirichlet_boundary([0, 0, 0]),
            'Reflective': anuga.Reflective_boundary(domain),
            'Transmissive': anuga.Transmissive_boundary(domain),
            'ghost': None
        }
        boundaries = dict()
        for tag in domain.boundary.values():
            boundaries[tag] = default_boundary_maps[tag]
        domain.set_boundary(boundaries)

        # setup rainfall
        def create_inflow_function(dataframe, name):
            def rain(time_in_seconds):
                t_sec = int(math.floor(time_in_seconds))
                return dataframe[name][t_sec]
            rain.__name__ = name
            return rain


        duration = input_data['scenario_config'].get('duration')
        date_rng = pd.date_range(start='1/1/1970', periods=duration + 1, freq='s')
        inflow_dataframe = pd.DataFrame(date_rng, columns=['datetime'])
        rainfall_inflow_polygons = [feature for feature in input_data.get('inflow').get('features') if feature.get('properties').get('type') == 'Rainfall']
        surface_inflow_lines = [feature for feature in input_data.get('inflow').get('features') if feature.get('properties').get('type') == 'Surface']
        catchment_polygons =  [feature for feature in input_data.get('catchment').get('features')] if input_data.get('catchment') else []
        boundary_polygon = input_data.get('boundary_polygon')
        for inflow_polygon in rainfall_inflow_polygons:
            polygon_name = inflow_polygon.get('id')
            inflow_dataframe[polygon_name] = float(inflow_polygon.get('properties').get('data'))
            inflow_function = create_inflow_function(inflow_dataframe, polygon_name)
            geometry = inflow_polygon.get('geometry').get('coordinates')
            Polygonal_rate_operator(domain, rate=inflow_function, factor=1.0e-6, polygon=geometry, default_rate=0.00)
        if len(rainfall_inflow_polygons) >= 1 and len(catchment_polygons) > 0:
            for catchment_polygon in catchment_polygons:
                uniform_rainfall_rate = float(rainfall_inflow_polygons[0].get('properties').get('data'))
                polygon_name = catchment_polygon.get('id')
                inflow_dataframe[polygon_name] = uniform_rainfall_rate
                inflow_function = create_inflow_function(inflow_dataframe, polygon_name)
                geometry = catchment_polygon.get('geometry').get('coordinates')[0]
                # The catchment needs to be wholly in the domain:
                if check_coordinates_are_in_polygon(geometry, boundary_polygon):
                    Polygonal_rate_operator(domain, rate=inflow_function, factor=-1.0e-6, polygon=geometry, default_rate=0.00)
        if len(rainfall_inflow_polygons) > 1 and len(catchment_polygons) > 0:
            raise NotImplementedError('Cannot handle multiple rainfall polygons together with catchment hydrology.')

        for inflow_line in surface_inflow_lines:
            polyline_name = inflow_line.get('id')
            inflow_dataframe[polyline_name] = float(inflow_line.get('properties').get('data'))
            inflow_function = create_inflow_function(inflow_dataframe, polyline_name)
            geometry = inflow_line.get('geometry').get('coordinates')
            # check that inflow line is actually in the domain:
            if check_coordinates_are_in_polygon(geometry, boundary_polygon):
                Inlet_operator(domain, geometry, Q=inflow_function)

        # Don't yield more than 100 timesteps into the SWW file, and smallest resolution is 60 seconds:
        yieldstep = 60 if math.floor(duration/100) < 60 else math.floor(duration/100)
        for t in domain.evolve(yieldstep=yieldstep, finaltime=duration):
            start = time.time()
            domain.write_time()
            # logger.info(f"domain.timestepping_statistics() on anuga.myid {anuga.myid}: {domain.timestepping_statistics()}")
            if anuga.myid == 0:
                percentage_done = str(round(t * 100 / duration, 0)).zfill(3)
                # logger.info(f"{domain.timestepping_statistics()}")
                update_web_interface(run_args, data={"status": f"{percentage_done}%"})
                stop = time.time()
                duration_minutes = round((stop - start) / 60, 2)
                logger.info(f'{percentage_done}% complete. Latest update took {duration_minutes} minutes.')
        barrier()
        domain.sww_merge(verbose=False, delete_old=True)
        barrier()

        if anuga.myid == 0:
            post_process_sww(package_dir, run_args=run_args)
            if run_args:
                result_zip_path = zip_result_package(package_dir, username, password, remove=False)
    except Exception as e:
        update_web_interface(run_args, data={'status': 'error'})
        logger.error(f"{traceback.format_exc()}")
        raise
    finally:
        finalize()
    logger.info(f"finished run: {input_data['run_label']}")
    return result_zip_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("username", nargs='?', help="your username(email) at hydrata.com", type=str)
    parser.add_argument("password", nargs='?', help="your password at hydrata.com", type=str)
    parser.add_argument("--package_dir", "-wd", help="the base directory for your simulation, it contains the scenario.json file", type=is_dir_check)
    args = parser.parse_args()
    username = args.username
    password = args.password
    package_dir = args.package_dir
    # logger.info(f'run.py got {package_dir}')
    if not package_dir:
        package_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    # logger.info(f'run.py using {package_dir}')
    try:
        run_sim(package_dir, username, password)
    except Exception as e:
        run_args = (package_dir, username, password)
        logger.exception(e, exc_info=True)
        update_web_interface(run_args, data={'status': 'error'})
        raise e
