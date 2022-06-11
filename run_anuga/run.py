import json

import numpy

import anuga
import argparse
import logging
import math
import os
import pandas as pd
import traceback

from logging import handlers
from osgeo import gdal

from anuga import distribute, finalize, barrier
from anuga.utilities import quantity_setting_functions as qs
from anuga.operators.rate_operators import Polygonal_rate_operator
from run_utils import is_dir_check, setup_input_data, update_web_interface, create_mesh, make_interior_holes_and_tags, \
    make_frictions, post_process_sww, zip_result_package

from celery.utils.log import get_task_logger
logger = get_task_logger(__name__)


def run_sim(package_dir, username=None, password=None):
    run_args = package_dir, username, password
    input_data = setup_input_data(package_dir)
    output_stats = dict()

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

    try:
        logger.info(f"{anuga.myid=}")
        if anuga.myid == 0:
            logger.info(f"update_web_interface - building mesh")
            update_web_interface(run_args, data={'status': 'building mesh'})
            anuga_mesh, mesher_mesh_filepath = create_mesh(input_data)
            logger.info(f"create_mesh")
            domain = anuga.shallow_water.shallow_water_domain.Domain(
                mesh_filename=input_data['mesh_filepath'],
                use_cache=False,
                verbose=True,
            )
            with open(mesher_mesh_filepath, 'r') as mesh_file:
                mesh_dict = json.load(mesh_file)

            mesh = mesh_dict['mesh']
            vertex = mesh_dict['vertex']
            vertices = numpy.array(vertex)
            elem = mesh['elem']
            points = vertices[:, :2]
            elev = vertices[:, 2]
            domain = anuga.Domain(
                points,
                elem,
                mesh_filename=input_data['mesh_filepath'],
                use_cache=False,
                verbose=True
            )
            domain.set_name(input_data['run_label'])
            domain.set_datadir(input_data['output_directory'])
            domain.set_minimum_storable_height(0.005)
            domain.set_quantity('elevation', elev, locations=vertices)
            frictions = make_frictions(input_data)
            friction_function = qs.composite_quantity_setting_function(
                frictions,
                domain
            )
            domain.set_quantity('friction', friction_function, verbose=True)
            domain.set_quantity('stage', 0.0, verbose=True)

            update_web_interface(run_args, data={'status': 'created mesh'})
        else:
            domain = None
        # logger.info(f"domain on anuga.myid {anuga.myid}: {domain}")
        barrier()
        domain = distribute(domain, verbose=True)
        # logger.info(f"domain on anuga.myid {anuga.myid} after distribute(): {domain}")
        default_boundary_maps = {
            'exterior': anuga.Dirichlet_boundary([0, 0, 0]),
            'interior': anuga.Reflective_boundary(domain),
            'Dirichlet': anuga.Dirichlet_boundary([0, 0, 0]),
            'dirichlet': anuga.Dirichlet_boundary([0, 0, 0]),
            'Reflective': anuga.Reflective_boundary(domain),
            'reflective': anuga.Reflective_boundary(domain),
            'Transmissive': anuga.Transmissive_boundary(domain),
            'transmissive': anuga.Transmissive_boundary(domain),
            'ghost': None
        }
        boundaries = dict()
        for tag in domain.boundary.values():
            boundaries[tag] = default_boundary_maps[tag]
        domain.set_boundary(boundaries)

        # setup rainfall
        def rain(time_in_seconds):
            t_sec = int(math.floor(time_in_seconds))
            return rain_df['rate_m_s'][t_sec]

        duration = input_data['scenario_config'].get('duration')
        # for testing, don't allow model runs longer than one hour
        duration = 60 * 60 if duration > 60 * 60 else duration
        constant_rainfall = input_data['scenario_config'].get('constant_rainfall') or 100
        date_rng = pd.date_range(start='1/1/2022', periods=duration + 1, freq='s')
        rain_df = pd.DataFrame(date_rng, columns=['datetime'])
        rain_df['rate_m_s'] = constant_rainfall / 1000
        Polygonal_rate_operator(domain, rate=rain, factor=1.0e-3, polygon=input_data['boundary_polygon'], default_rate=0.00)

        # Don't yield more than 1000 timesteps into the SWW file, and smallest resolution is 60 seconds:
        yieldstep = 60 if math.floor(duration/1000) < 60 else math.floor(duration/1000)
        for t in domain.evolve(yieldstep=yieldstep, finaltime=duration):
            domain.write_time()
            # logger.info(f"domain.timestepping_statistics() on anuga.myid {anuga.myid}: {domain.timestepping_statistics()}")
            if anuga.myid == 0:
                logger.info(f'domain.evolve {t} on processor {anuga.myid}')
                # logger.info(f"{domain.timestepping_statistics()}")
                update_web_interface(run_args, data={"status": f"{round(t/duration * 100, 0)}%"})
        barrier()
        domain.sww_merge(verbose=True, delete_old=True)
        barrier()

        if anuga.myid == 0:
            post_process_sww(package_dir, run_args=run_args)
            if run_args:
                zip_result_package(package_dir, username, password, remove=False)
    except Exception as e:
        update_web_interface(run_args, data={'status': 'error'})
        logger.error(f"{traceback.format_exc()}")
    finally:
        finalize()
    logger.info(f"finished run: {input_data['run_label']}")
    return f"finished run: {input_data['run_label']}"


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
