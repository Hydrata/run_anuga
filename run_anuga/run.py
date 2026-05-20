import argparse
import json
import logging
import math
import os
import signal
import time
import traceback

from run_anuga._imports import import_optional
from run_anuga.run_utils import is_dir_check, setup_input_data, update_web_interface, create_mesher_mesh, create_anuga_mesh, \
    make_frictions, post_process_sww, setup_logger, RunContext, \
    build_time_boundary_function, apply_inflows_to_domain
from run_anuga import defaults
from run_anuga.callbacks import NullCallback, HydrataCallback

try:
    from celery.utils.log import get_task_logger
    logger = get_task_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


# SIGALRM watchdog around MPI_Finalize — defends against the libmpi
# ompi_mpi_finalize → usleep busy-loop wedge observed in run 27593 forensics.
def _finalize_with_timeout(finalize, timeout_seconds=None):
    if timeout_seconds is None:
        timeout_seconds = int(os.environ.get('RUN_ANUGA_FINALIZE_TIMEOUT_SECONDS', '30'))

    def _on_timeout(signum, frame):
        raise TimeoutError(f"MPI_Finalize exceeded {timeout_seconds}s")

    previous_handler = signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(timeout_seconds)
    try:
        finalize()
    except TimeoutError:
        logger.warning(
            f"MPI_Finalize hung after {timeout_seconds}s and was abandoned; "
            "the run is otherwise complete and the OS will reclaim MPI state on process exit."
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def run_sim(package_dir, username=None, password=None, batch_number=1, checkpoint_time=None, callback=None):
    # Lazy imports — these are only needed when actually running a simulation.
    anuga = import_optional("anuga")
    pickle = import_optional("dill")
    numpy = import_optional("numpy")
    psutil = import_optional("psutil")
    from anuga import distribute, finalize, barrier, Inlet_operator
    from anuga.utilities import quantity_setting_functions as qs
    from anuga.operators.rate_operators import Polygonal_rate_operator

    # Keep run_args for backward compat with update_web_interface in main() error handler.
    run_args = RunContext(package_dir, username, password)
    input_data = setup_input_data(package_dir)

    # Backward compat: auto-construct callback from username/password if not provided.
    if callback is None and username:
        callback = HydrataCallback.from_config(
            username, password, input_data['scenario_config']
        )
    callback = callback or NullCallback()
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
            logger.info("Building domain...")
            sub_domain_name = None
            if anuga.numprocs > 1:
                sub_domain_name = domain_name + "_P{}_{}".format(anuga.numprocs, anuga.myid)
            pickle_name = (os.path.join(checkpoint_directory, sub_domain_name) + "_" + str(checkpoint_time) + ".pickle")
            try:
                domain = pickle.load(open(pickle_name, "rb"))
                logger.info(f"{pickle_name=}")
                success = True
            except Exception:
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
            callback.on_status('created mesh')
            logger.info(domain.mesh.statistics())
        else:
            domain = None
        if batch_number == 1:
            barrier()
            domain = distribute(domain, verbose=True)

            # Inflow.make_file() resolved each feature's `data` via
            # FeatureDataMixin (TASK-820). Per-feature shapes (None / list /
            # float) are handled inside apply_inflows_to_domain, which is
            # unit-tested at tests/unit/test_run_anuga/test_apply_inflows.py.
            apply_inflows_to_domain(
                input_data=input_data,
                domain=domain,
                start=start,
                duration=duration,
                Polygonal_rate_operator=Polygonal_rate_operator,
                Inlet_operator=Inlet_operator,
                defaults_module=defaults,
            )
        default_boundary_maps = {
            'exterior': anuga.Dirichlet_boundary([0, 0, 0]),
            'interior': anuga.Reflective_boundary(domain),
            'Dirichlet': anuga.Dirichlet_boundary([0, 0, 0]),
            'Reflective': anuga.Reflective_boundary(domain),
            'Transmissive': anuga.Transmissive_boundary(domain),
            'ghost': None
        }
        # Build a 'Time' boundary entry only when at least one external
        # boundary feature carries boundary='Time'. The per-feature `data`
        # has already been resolved server-side by Boundary.make_file —
        # it arrives here as either a numeric stage value (constant case)
        # or a list of {timestamp, value} dicts (TimeSeries case).
        time_boundary_features = [
            f for f in (input_data.get('boundary', {}).get('features') or [])
            if (f.get('properties') or {}).get('location') == 'External'
            and (f.get('properties') or {}).get('boundary') == 'Time'
        ]
        if time_boundary_features:
            time_function = build_time_boundary_function(
                time_boundary_features, defaults
            )
            default_boundary_maps['Time'] = anuga.Time_boundary(
                domain=domain, function=time_function,
            )
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
        # W6 (TASK-1044) — `simulation_start` is the absolute wall-clock anchor used
        # for ETA estimation; `start` is the per-tick reference reset every iteration.
        simulation_start = time.time()
        start = simulation_start
        for t in domain.evolve(yieldstep=yieldstep, finaltime=duration, skip_initial_step=skip_initial_step):
            if anuga.myid == 0:
                stop = time.time()
                percentage_done = round(t * 100 / duration, 1)
                # W6 (TASK-1044) — switch numeric progress from on_status('X%') to
                # on_progress(X). on_status is reserved for state words ('error' below
                # stays). ETA = elapsed * (100 - pct) / pct; unknown when pct==0.
                elapsed = stop - simulation_start
                if percentage_done > 0:
                    eta_seconds = int(elapsed * (100 - percentage_done) / percentage_done)
                else:
                    eta_seconds = None
                callback.on_progress(percentage_done, eta_seconds=eta_seconds)
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
            callback.on_metric('memory_used', max_memory_usage)
            logger.info("Processing results...")
            post_process_sww(package_dir, run_args=run_args)
    except Exception:
        callback.on_status('error')
        logger.error(f"{traceback.format_exc()}")
        raise
    finally:
        _finalize_with_timeout(finalize)
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
        run_args = RunContext(package_dir, username, password)
        logger.exception("run.py main() failed")
        _report_run_error(run_args, str(e))
        raise e


def _report_run_error(run_args, message):
    """POST the run failure to the dedicated /error/ endpoint.

    The endpoint calls Run.mark_error() server-side, which appends to the
    run log, mirrors onto any linked Compute row, and is idempotent on
    already-terminal runs. Failures here are logged but never raised — we
    must not mask the originating exception.
    """
    try:
        package_dir = run_args.package_dir
        username = run_args.username
        password = run_args.password
        if not (username and password):
            return
        scenario_json_path = os.path.join(package_dir, 'scenario.json')
        with open(scenario_json_path, 'r') as f:
            scenario_config = json.load(f)
        run_id = scenario_config.get('run_id')
        control_server = scenario_config.get('control_server')
        if not (control_server and run_id):
            return
        requests = import_optional("requests")
        from run_anuga._http import post_to_control_server

        url = f"{control_server}api/v2/anuga/runs/{run_id}/error/"
        auth = requests.auth.HTTPBasicAuth(username, password)
        # Small POST with a scalar message: a 30s upper bound is fine here
        # (the helper default is None / no timeout, which is required for the
        # PATCH-with-files callers but inappropriate for an error report).
        post_to_control_server(
            url, auth=auth, method="POST", data={'message': message}, timeout=30,
        )
    except Exception:
        logger.exception("Failed to report run error to control server")


if __name__ == '__main__':
    main()
