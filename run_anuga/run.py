import argparse
import json
import logging
import math
import os
import signal
import time
import traceback
from datetime import datetime

from run_anuga._imports import import_optional
from run_anuga.run_utils import is_dir_check, setup_input_data, update_web_interface, create_mesher_mesh, create_anuga_mesh, \
    make_frictions, post_process_sww, check_coordinates_are_in_polygon, compute_yieldstep, RunContext
from run_anuga import defaults
from run_anuga.callbacks import NullCallback, HydrataCallback
from run_anuga.diagnostics import SimulationMonitor
from run_anuga.logging_setup import configure_simulation_logging, neutralize_anuga_logging, teardown_simulation_logging

_module_logger = logging.getLogger(__name__)


def run_sim(package_dir, username=None, password=None, batch_number=1, checkpoint_time=None, callback=None):
    # Early check: fail fast with a friendly message if ANUGA isn't installed.
    import_optional("anuga")

    # --- Phase 1: parse scenario (needs shapely but not ANUGA domain objects) ---
    input_data = setup_input_data(package_dir)
    batch_number = int(batch_number)

    # --- Phase 2: configure logging ---
    logger = configure_simulation_logging(input_data['output_directory'], batch_number)

    # --- Phase 3: lazy-import ANUGA ---
    anuga = import_optional("anuga")
    pickle = import_optional("dill")
    numpy = import_optional("numpy")
    psutil = import_optional("psutil")
    pd = import_optional("pandas")
    from anuga import distribute, finalize, barrier, Inlet_operator
    from anuga.utilities import quantity_setting_functions as qs
    from anuga.operators.rate_operators import Polygonal_rate_operator

    # --- Phase 4: neutralize ANUGA's logging before any log.*() call ---
    neutralize_anuga_logging(input_data['output_directory'])

    # Backward compat: auto-construct callback from username/password if not provided.
    if callback is None and username:
        callback = HydrataCallback.from_config(
            username, password, input_data['scenario_config']
        )
    callback = callback or NullCallback()
    logger.info(f"run_sim started with {batch_number=}")
    domain = None
    memory_usage_logs = list()
    duration = input_data['scenario_config'].get('duration')
    model_start = input_data['scenario_config'].get('model_start')
    # Absolute domain starttime in seconds (Unix epoch). Used to compute finaltime
    # and normalise progress/diagnostics to simulation-relative time (0…duration).
    if model_start:
        starttime_s = int(datetime.fromisoformat(model_start.replace('Z', '+00:00')).timestamp())
    else:
        starttime_s = 0
    try:
        domain_name = input_data['run_label']
        checkpoint_directory = input_data['checkpoint_directory']
        if batch_number > 1:
            overall = None
            logger.info("Building domain...")
            sub_domain_name = None
            if anuga.numprocs > 1:
                sub_domain_name = domain_name + "_P{}_{}".format(anuga.numprocs, anuga.myid)
            pickle_name = (os.path.join(checkpoint_directory, sub_domain_name) + "_" + str(checkpoint_time) + ".pickle")
            try:
                domain = pickle.load(open(pickle_name, "rb"))
                logger.debug(f"{pickle_name=}")
                success = True
            except Exception:
                success = False
            for attempt in range(5):
                logger.debug(f"overall attempt: {attempt}")
                overall = success
                for cpu in range(anuga.numprocs):
                    if cpu != anuga.myid:
                        anuga.send(success, cpu)
                        if attempt > 1:
                            logger.debug(f"cpu sent: {cpu}, {success}")
                for cpu in range(anuga.numprocs):
                    if cpu != anuga.myid:
                        result = anuga.receive(cpu)
                        if isinstance(result, tuple):  # If result is a tuple, then it is (buffer, rs)
                            buffer, rs = result
                        else:  # If result is not a tuple, then it is just buffer
                            buffer = result
                            rs = None  # Or some other default value, as per your requirement
                        if attempt > 1:
                            logger.debug(f"cpu receive: {cpu}: {buffer}, {rs}, attempt: {attempt}")
                        overall = overall & buffer
                logger.debug(f"{overall=}")
                if overall:
                    break
                time.sleep(10)
            domain.set_evolve_starttime(checkpoint_time)
            barrier()
            if not overall:
                raise Exception(f"Unable to open checkpoint file: {pickle_name}")
            domain.last_walltime = time.time()
            domain.communication_time = 0.0
            domain.communication_reduce_time = 0.0
            domain.communication_broadcast_time = 0.0
            logger.info('load_checkpoint_file succeeded. Checkpoint domain set.')
        elif anuga.myid == 0:
            logger.info("Building domain...")
            logger.info('No checkpoint file found. Starting new Simulation')
            callback.on_status('building mesh')
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
                    callback.on_metric('mesh_triangle_count', mesh_size)
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
            if model_start:
                domain.set_starttime(starttime_s)
            flow_algorithm = input_data['scenario_config'].get('flow_algorithm')
            if flow_algorithm:
                logger.info(f"Setting flow algorithm: {flow_algorithm}")
                domain.set_flow_algorithm(flow_algorithm)
                if flow_algorithm.endswith('_SG') and input_data.get('elevation_filename'):
                    logger.info(f"Building sub-grid tables from {input_data['elevation_filename']}")
                    domain.set_subgrid_dem(input_data['elevation_filename'], verbose=True)
            callback.on_status('created mesh')
            logger.info(domain.mesh.statistics())
        else:
            domain = None
        if batch_number == 1:
            barrier()
            domain = distribute(domain, verbose=True)
            domain.optimise_dry_cells = True  # Skip reconstruction for dry cells — free speedup, no accuracy impact

            # setup rainfall
            def create_inflow_function(dataframe, name):
                def rain(time_in_seconds):
                    # Normalise to simulation-relative seconds (0…duration).
                    # starttime_s=0 when model_start is unset; equals Unix epoch otherwise.
                    # Clamped so ANUGA's function-type probe (called with t=0) stays in bounds.
                    t_sec = max(0, min(int(math.floor(time_in_seconds - starttime_s)), duration))
                    return dataframe[name].iloc[t_sec]

                rain.__name__ = name
                return rain

            inflow_features = (input_data.get('inflow') or {}).get('features') or []
            rainfall_inflow_polygons = [f for f in inflow_features if f.get('properties').get('type') == 'Rainfall']
            surface_inflow_lines = [f for f in inflow_features if f.get('properties').get('type') == 'Surface']
            catchment_polygons = [feature for feature in input_data.get('catchment').get('features')] if input_data.get(
                'catchment') else []
            boundary_polygon = input_data.get('boundary_polygon')
            datetime_range = pd.date_range(start=model_start or '1/1/1970', periods=duration + 1, freq='s')
            inflow_dataframe = pd.DataFrame(datetime_range, columns=['timestamp'])
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
                geometry = inflow_polygon.get('geometry').get('coordinates')[0]
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
        yieldstep = compute_yieldstep(duration)
        checkpoint_directory = input_data['checkpoint_directory']
        domain.set_checkpointing(
            checkpoint=True,
            checkpoint_dir=checkpoint_directory,
            checkpoint_step=1
        )
        barrier()
        monitor = None
        if anuga.myid == 0:
            monitor = SimulationMonitor(
                domain,
                input_data['output_directory'],
                batch_number,
                yieldstep,
                duration_s=duration,
                run_label=input_data['run_label'],
                scenario_config=input_data['scenario_config'],
            )

        # --- Graceful bail-out mechanism ---
        # Send SIGUSR1 to rank-0 PID to request a clean stop at the next yieldstep.
        # The checkpoint written at the previous yieldstep can be used to resume:
        #   run-anuga run <dir> --batch_number 2 --checkpoint_time <t>
        _bail_out = False
        _bail_flag_path = os.path.join(input_data['output_directory'], 'bail.flag')

        def _handle_sigusr1(sig, frame):
            nonlocal _bail_out
            _bail_out = True
            try:
                with open(_bail_flag_path, 'w') as _f:
                    _f.write(f"bail requested at {time.time()}\n")
            except Exception:
                pass

        if anuga.myid == 0 and hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, _handle_sigusr1)

        percentage_done = 0.0
        _yieldstep_start = time.time()
        for t in domain.evolve(yieldstep=yieldstep, finaltime=starttime_s + duration):
            # All ranks check the bail flag file (written by rank-0 signal handler)
            if os.path.exists(_bail_flag_path):
                _bail_out = True

            if anuga.myid == 0:
                wall_time_s = time.time() - _yieldstep_start
                percentage_done = round((t - starttime_s) * 100 / duration, 1)
                callback.on_status(f"{percentage_done}%")
                duration_seconds = round(wall_time_s)
                minutes, seconds = divmod(duration_seconds, 60)
                memory_percent = psutil.virtual_memory().percent
                memory_usage = psutil.virtual_memory().used
                mem_mb = memory_usage / (1024 * 1024)
                memory_usage_logs.append(memory_usage)
                diag = monitor.record(t - starttime_s, wall_time_s=wall_time_s, mem_mb=mem_mb)
                log_fn = logger.info
                mem_note = ''
                if memory_percent >= 92:
                    log_fn = logger.critical
                    mem_note = ' *** MEMORY CRITICALLY LOW — send SIGUSR1 to rank-0 for graceful checkpoint+exit ***'
                elif memory_percent >= 85:
                    log_fn = logger.warning
                    mem_note = ' *** memory pressure high ***'
                log_fn(
                    f'{percentage_done}% | {minutes}m {seconds}s | '
                    f'mem: {memory_percent}% ({mem_mb:.0f} MB) | disk: {psutil.disk_usage("/").percent}% | '
                    f'{domain.get_datetime().isoformat()} | '
                    + monitor.format_log_suffix(diag)
                    + mem_note
                )
                _yieldstep_start = time.time()

            if _bail_out:
                if anuga.myid == 0:
                    logger.warning(
                        f"Bail signal received at t={t - starttime_s:.1f}s "
                        f"({percentage_done}% complete) — "
                        f"checkpoint saved, exiting cleanly. "
                        f"Resume: --batch_number {batch_number + 1} --checkpoint_time {int(t)}"
                    )
                break

        bail_note = ' (bailed early)' if _bail_out else ''
        barrier()
        domain.sww_merge(verbose=not _bail_out, delete_old=True)
        barrier()
        if anuga.myid == 0:
            monitor.finalize()
            max_memory_usage = int(round(max(memory_usage_logs))) if memory_usage_logs else 0
            callback.on_metric('memory_used', max_memory_usage)
            if _bail_out:
                logger.warning(f"Simulation stopped early{bail_note}. SWW files merged. "
                                f"Restart with --batch_number {batch_number + 1} --checkpoint_time {int(t)}")
                # Clean up bail flag so a fresh restart doesn't immediately bail again
                if os.path.exists(_bail_flag_path):
                    os.remove(_bail_flag_path)
                return
            logger.info("Processing results...")
            try:
                post_process_sww(package_dir)
            except Exception:
                logger.warning("post_process_sww failed (TIF output skipped):\n%s", traceback.format_exc())

            # Log output file paths
            sww_path = os.path.join(input_data['output_directory'], f"{input_data['run_label']}.sww")
            if os.path.isfile(sww_path):
                sww_size_mb = os.path.getsize(sww_path) / (1024 * 1024)
                logger.info(f"SWW output: {sww_path} ({sww_size_mb:.1f} MB)")
            tif_files = [f for f in os.listdir(input_data['output_directory']) if f.endswith('.tif')]
            logger.info(f"Output directory: {input_data['output_directory']} ({len(tif_files)} .tif files)")
            log_path = os.path.join(input_data['output_directory'], f"run_anuga_{batch_number}.log")
            logger.info(f"Log file: {log_path}")
        logger.info(f"finished run: {input_data['run_label']}")
    except Exception:
        callback.on_status('error')
        logger.error(f"{traceback.format_exc()}")
        raise
    finally:
        teardown_simulation_logging()
        finalize()


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
        _module_logger.debug(f"run.py main() running {batch_number=}")
        run_sim(package_dir, username, password, batch_number, checkpoint_time)
    except Exception as e:
        run_args = RunContext(package_dir, username, password)
        _module_logger.exception(e, exc_info=True)
        update_web_interface(run_args, data={'status': 'error'})
        raise e


if __name__ == '__main__':
    main()
