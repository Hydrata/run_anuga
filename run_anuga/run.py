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
from anuga.utilities import plot_utils as util
from anuga.operators.rate_operators import Polygonal_rate_operator
from run_utils import is_dir_check, setup_input_data, update_web_interface, create_mesh, make_interior_holes_and_tags, \
    make_frictions

logger = logging.getLogger(__name__)


def run_sim(package_dir, username=None, password=None):
    run_args = package_dir, username, password
    input_data = setup_input_data(package_dir)
    output_stats = dict()

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(input_data['output_directory'], 'run_anuga.log'))
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    console_format = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    file_format = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)

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
        web_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        web_handler.setFormatter(web_format)
        logger.addHandler(web_handler)

    try:
        if anuga.myid == 0:
            update_web_interface(run_args, data={'status': 'building mesh'})
            mesh = create_mesh(input_data)
            domain = anuga.shallow_water.shallow_water_domain.Domain(
                mesh_filename=input_data['mesh_filepath'],
                use_cache=False,
                verbose=True,
            )
            domain.set_name(input_data['run_label'])
            domain.set_datadir(input_data['output_directory'])
            domain.set_minimum_storable_height(0.005)

            poly_fun_pairs = [['Extent', input_data['elevation_filename']]]
            elevation_function = qs.composite_quantity_setting_function(
                poly_fun_pairs,
                domain,
                nan_treatment='exception',
            )
            domain.set_quantity('elevation', elevation_function, verbose=True, alpha=0.99, location='centroids')
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
        barrier()
        domain = distribute(domain, verbose=True)
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

        for t in domain.evolve(yieldstep=60, finaltime=duration):
            domain.write_time()
            if anuga.myid == 0:
                logger.info(f'domain.evolve {t} on processor {anuga.myid}')
                logger.info(f"{domain.timestepping_statistics()}")
                update_web_interface(run_args, data={"status": f"{round(t/duration * 100, 0)}%"})
        domain.sww_merge(verbose=True, delete_old=True)
        barrier()

        if anuga.myid == 0:
            logger.info('Generating output rasters...')
            raster = gdal.Open(input_data['elevation_filename'])
            gt = raster.GetGeoTransform()
            resolution = 1 if math.floor(gt[1] / 4) == 0 else math.floor(gt[1] / 4)
            resolutions = list()
            for feature in input_data.get('mesh_region').get('features') or list():
                logger.info(f'{feature=}')
                resolutions.append(feature.get('properties').get('resolution'))
            logger.info(f'{resolutions=}')
            if len(resolutions) == 0:
                resolutions = [1000]
            highest_grid_resolution = math.floor(math.sqrt(2 * min(resolutions)))
            logger.info(f'{highest_grid_resolution=}')
            logger.info(f'raster resolution: {highest_grid_resolution}m')
            epsg_integer = int(input_data['scenario_config'].get("epsg").split(":")[1]
                               if ":" in input_data['scenario_config'].get("epsg")
                               else input_data['scenario_config'].get("epsg"))
            interior_holes, _ = make_interior_holes_and_tags(input_data)
            logger.info(f'{interior_holes=}')
            util.Make_Geotif(
                swwFile=f"{input_data['output_directory']}/{input_data['run_label']}.sww",
                output_quantities=['depth', 'velocity', 'depthIntegratedVelocity'],
                myTimeStep='max',
                CellSize=highest_grid_resolution,
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
    except Exception as e:
        logger.error(f"{traceback.format_exc()}")
    finally:
        barrier()
        finalize()
    return f"finished: {input_data['run_label']}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("username", nargs='?', help="your username(email) at hydrata.com", type=str)
    parser.add_argument("password", nargs='?', help="your password at hydrata.com", type=str)
    parser.add_argument("--package_dir", "-wd", help="the base directory for your simulation, it contains the scenario.json file", type=is_dir_check)
    args = parser.parse_args()
    username = args.username
    password = args.password
    package_dir = args.package_dir
    logger.info(f'run.py got {package_dir}')
    if not package_dir:
        package_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    logger.info(f'run.py using {package_dir}')
    try:
        run_sim(package_dir, username, password)
    except Exception as e:
        run_args = (package_dir, username, password)
        logging.exception(e, exc_info=True)
        update_web_interface(run_args, data={'status': 'error'})
        raise e
