import json
import dill as pickle
import time
import numpy
import psutil
import anuga
import argparse
import math
import os
import pandas as pd
import traceback

from anuga import distribute, finalize, barrier, Inlet_operator
from anuga.utilities import quantity_setting_functions as qs
from anuga.operators.rate_operators import Polygonal_rate_operator

from run_anuga.run_utils import is_dir_check, setup_input_data, update_web_interface, create_mesher_mesh, create_anuga_mesh, \
    make_frictions, post_process_sww, setup_logger, check_coordinates_are_in_polygon
from run_anuga import defaults

try:
    from celery.utils.log import get_task_logger
    logger = get_task_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def run_sim(package_dir, username=None, password=None, batch_number=1, checkpoint_time=None):
    run_args = package_dir, username, password
    input_data = setup_input_data(package_dir)
    logger = setup_logger(input_data, username, password, batch_number)
    logger.info(f"run_sim started with {batch_number=}")
    domain = None
    overall = None
    skip_initial_step = False
    memory_usage_logs = list()
    duration = input_data['scenario_config'].get('duration')
    start = '1/1/1970'
    batch_number = int(batch_number)
    if input_data['scenario_config'].get('model_start'):
        start = input_data['scenario_config'].get('model_start')
    try:
        domain_name = input_data['run_label']
        checkpoint_directory = input_data['checkpoint_directory']
        if batch_number > 1:
            logger.info(f"Building domain...")
            sub_domain_name = None
            if anuga.numprocs > 1:
                sub_domain_name = domain_name + "_P{}_{}".format(anuga.numprocs, anuga.myid)
            pickle_name = (os.path.join(checkpoint_directory, sub_domain_name) + "_" + str(checkpoint_time) + ".pickle")
            try:
                domain = pickle.load(open(pickle_name, "rb"))
                logger.info(f"{pickle_name=}")
                success = True
            except:
                success = False
            for attempt in range(5):
                logger.info(f"overall attempt: {attempt}")
                overall = success
                for cpu in range(anuga.numprocs):
                    if cpu != anuga.myid:
                        anuga.send(success, cpu)
                        if attempt > 1:
                            logger.info(f"cpu sent: {cpu}, {success}")
                for cpu in range(anuga.numprocs):
                    if cpu != anuga.myid:
                        result = anuga.receive(cpu)
                        if isinstance(result, tuple):  # If result is a tuple, then it is (buffer, rs)
                            buffer, rs = result
                        else:  # If result is not a tuple, then it is just buffer
                            buffer = result
                            rs = None  # Or some other default value, as per your requirement
                        if attempt > 1:
                            logger.info(f"cpu receive: {cpu}: {buffer}, {rs}, attempt: {attempt}")
                        overall = overall & buffer
                logger.info(f"{overall=}")
                time.sleep(10)
                if overall:
                    break
            domain.set_evolve_starttime(checkpoint_time)
            barrier()
            if not overall:
                raise Exception(f"Unable to open checkpoint file: {pickle_name}")
            domain.last_walltime = time.time()
            # skip_initial_step = True
            domain.communication_time = 0.0
            domain.communication_reduce_time = 0.0
            domain.communication_broadcast_time = 0.0
            logger.info('load_checkpoint_file succeeded. Checkpoint domain set.')
        elif anuga.myid == 0:
            logger.info(f"Building domain...")
            logger.info('No checkpoint file found. Starting new Simulation')
            update_web_interface(run_args, data={'status': 'building mesh'})
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
                if not os.path.isfile(input_data['mesh_filepath']):
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
            frictions = make_frictions(input_data)
            friction_function = qs.composite_quantity_setting_function(
                frictions,
                domain
            )
            domain.set_quantity('friction', friction_function, verbose=False)
            domain.set_quantity('stage', 0.0, verbose=False)
            domain.set_name(input_data['run_label'])
            domain.set_datadir(input_data['output_directory'])
            domain.set_minimum_storable_height(defaults.MINIMUM_STORABLE_HEIGHT_M)
            update_web_interface(run_args, data={'status': 'created mesh'})
            logger.info(domain.mesh.statistics())
        else:
            domain = None
        if batch_number == 1:
            barrier()
            domain = distribute(domain, verbose=True)

            # setup rainfall
            def create_inflow_function(dataframe, name):
                def rain(time_in_seconds):
                    t_sec = int(math.floor(time_in_seconds))
                    return dataframe[name][t_sec]

                rain.__name__ = name
                return rain

            rainfall_inflow_polygons = [feature for feature in input_data.get('inflow').get('features') if
                                        feature.get('properties').get('type') == 'Rainfall']
            surface_inflow_lines = [feature for feature in input_data.get('inflow').get('features') if
                                    feature.get('properties').get('type') == 'Surface']
            catchment_polygons = [feature for feature in input_data.get('catchment').get('features')] if input_data.get(
                'catchment') else []
            boundary_polygon = input_data.get('boundary_polygon')
            datetime_range = pd.date_range(start=start, periods=duration + 1, freq='s')
            inflow_dataframe = pd.DataFrame(datetime_range, columns=['timestamp'])
            inflow_functions = dict()
            for inflow_polygon in rainfall_inflow_polygons:
                polygon_name = inflow_polygon.get('id')
                data = inflow_polygon.get('properties').get('data')
                if isinstance(data, list):
                    new_dataframe = pd.DataFrame(data)
                    new_dataframe['timestamp'] = pd.to_datetime(new_dataframe['timestamp'])
                    new_dataframe[polygon_name] = pd.to_numeric(new_dataframe['value'])
                    if inflow_dataframe['timestamp'].dt.tz is None:
                        inflow_dataframe['timestamp'] = inflow_dataframe['timestamp'].dt.tz_localize('UTC')
                    if new_dataframe['timestamp'].dt.tz is None:
                        new_dataframe['timestamp'] = new_dataframe['timestamp'].dt.tz_localize('UTC')
                    inflow_dataframe = pd.merge(inflow_dataframe, new_dataframe, how='left', on='timestamp')
                    inflow_dataframe.ffill(inplace=True)
                else:
                    inflow_dataframe[polygon_name] = float(data)
                inflow_function = create_inflow_function(inflow_dataframe, polygon_name)
                inflow_functions[polygon_name] = inflow_function
                geometry = inflow_polygon.get('geometry').get('coordinates')
                Polygonal_rate_operator(domain, rate=inflow_function, factor=defaults.RAINFALL_FACTOR, polygon=geometry,
                                        default_rate=0.00)
            if len(rainfall_inflow_polygons) >= 1 and len(catchment_polygons) > 0:
                for catchment_polygon in catchment_polygons:
                    uniform_rainfall_rate = float(rainfall_inflow_polygons[0].get('properties').get('data'))
                    polygon_name = catchment_polygon.get('id')
                    inflow_dataframe[polygon_name] = uniform_rainfall_rate
                    inflow_function = create_inflow_function(inflow_dataframe, polygon_name)
                    geometry = catchment_polygon.get('geometry').get('coordinates')[0]
                    # The catchment needs to be wholly in the domain:
                    if check_coordinates_are_in_polygon(geometry, boundary_polygon):
                        Polygonal_rate_operator(domain, rate=inflow_function, factor=-defaults.RAINFALL_FACTOR, polygon=geometry,
                                                default_rate=0.00)
            if len(rainfall_inflow_polygons) > 1 and len(catchment_polygons) > 0:
                raise NotImplementedError('Cannot handle multiple rainfall polygons together with catchment hydrology.')

            for inflow_line in surface_inflow_lines:
                polyline_name = inflow_line.get('id')
                inflow_dataframe[polyline_name] = float(inflow_line.get('properties').get('data'))
                inflow_function = create_inflow_function(inflow_dataframe, polyline_name)
                inflow_functions[polyline_name] = inflow_function
                geometry = inflow_line.get('geometry').get('coordinates')
                # check that inflow line is actually in the domain:
                if check_coordinates_are_in_polygon(geometry, boundary_polygon):
                    Inlet_operator(domain, geometry, Q=inflow_function)
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
        max_yieldsteps = defaults.MAX_YIELDSTEPS
        temporal_resolution_seconds = defaults.MIN_YIELDSTEP_S
        base_temporal_resolution_seconds = math.floor(duration/max_yieldsteps)
        yieldstep = base_temporal_resolution_seconds
        if base_temporal_resolution_seconds < temporal_resolution_seconds:
            yieldstep = temporal_resolution_seconds
        if yieldstep > defaults.MAX_YIELDSTEP_S:
            yieldstep = defaults.MAX_YIELDSTEP_S
        checkpoint_directory = input_data['checkpoint_directory']
        domain.set_checkpointing(
            checkpoint=True,
            checkpoint_dir=checkpoint_directory,
            checkpoint_step=1
        )
        barrier()
        start = time.time()
        for t in domain.evolve(yieldstep=yieldstep, finaltime=duration, skip_initial_step=skip_initial_step):
            if anuga.myid == 0:
                stop = time.time()
                percentage_done = round(t * 100 / duration, 1)
                update_web_interface(run_args, data={"status": f"{percentage_done}%"})
                duration_seconds = round(stop - start)
                minutes, seconds = divmod(duration_seconds, 60)
                memory_percent = psutil.virtual_memory().percent
                memory_usage = psutil.virtual_memory().used
                memory_usage_logs.append(memory_usage)
                logger.info(f'{percentage_done}% | {minutes}m {seconds}s | mem: {memory_percent}% | disk: {psutil.disk_usage("/").percent}% | {domain.get_datetime().isoformat()}')
                start = time.time()
        barrier()
        domain.sww_merge(verbose=True, delete_old=True)
        barrier()
        if anuga.myid == 0:
            max_memory_usage = int(round(max(memory_usage_logs)))
            update_web_interface(run_args, data={"memory_used": max_memory_usage})
            logger.info("Processing results...")
            post_process_sww(package_dir, run_args=run_args)
    except Exception as e:
        update_web_interface(run_args, data={'status': 'error'})
        logger.error(f"{traceback.format_exc()}")
        raise
    finally:
        finalize()
    logger.info(f"finished run: {input_data['run_label']}")


def main():
    parser = argparse.ArgumentParser(description="Run an ANUGA flood simulation from a Hydrata scenario package.")
    parser.add_argument("username", nargs='?', help="your username(email) at hydrata.com", type=str)
    parser.add_argument("password", nargs='?', help="your password at hydrata.com", type=str)
    parser.add_argument("--package_dir", "-pd", help="the base directory for your simulation, it contains the scenario.json file", type=is_dir_check)
    parser.add_argument("--batch_number", "-bn", help="when using checkpointing, the batch_number, is the number of times the run has been restarted.", type=str)
    parser.add_argument("--checkpoint_time", "-ct", help="when using checkpointing, the checkpoint_time, is the time in seconds, to restart the simulation from.", type=str)
    args = parser.parse_args()
    username = args.username
    password = args.password
    package_dir = args.package_dir
    batch_number = args.batch_number
    checkpoint_time = args.checkpoint_time
    if not package_dir:
        package_dir = os.path.join(os.path.dirname(__file__), '..')
    try:
        logger.info(f"run.py main() running {batch_number=}")
        run_sim(package_dir, username, password, batch_number, checkpoint_time)
    except Exception as e:
        run_args = (package_dir, username, password)
        logger.exception(e, exc_info=True)
        update_web_interface(run_args, data={'status': 'error'})
        raise e


if __name__ == '__main__':
    main()
