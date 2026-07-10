import argparse
import json
import logging
import os
import signal
import sys
import time
import traceback

from run_anuga._imports import import_optional
from run_anuga.run_utils import is_dir_check, setup_input_data, create_anuga_mesh, \
    make_frictions, post_process_sww, setup_logger, RunContext, \
    build_time_boundary_function, apply_inflows_to_domain, \
    assert_raster_has_no_nodata_inside_boundary, make_raised_elevation_pairs, \
    apply_raised_elevation_correction, \
    compute_mesh_qa, extract_boundary_condition_types, compute_yieldstep  # W3 (TASK-1923)
from run_anuga import defaults
from run_anuga import phase_tracker
from run_anuga.callbacks import NullCallback, HydrataCallback
from run_anuga.diagnostics import SimulationMonitor, finalize_monitor_safely
from run_anuga._logging import install_mname_filter

try:
    from celery.utils.log import get_task_logger
    logger = get_task_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# Stamp anuga_core's mname/lnum record fields so run_anuga logs format cleanly
# when they propagate to anuga's root %(mname)s formatter (TASK-1276).
install_mname_filter(logger)

# celery's get_task_logger returns a logger with no stdout handler outside a
# celery worker (which is the case in the Batch container), so logger.error()
# in the rank-0 broad-except would otherwise be silently dropped from
# CloudWatch. Idempotent: skip if any caller already attached one.
if not any(
    isinstance(h, logging.StreamHandler) and getattr(h, 'stream', None) is sys.stderr
    for h in logger.handlers
):
    _stderr_handler = logging.StreamHandler(sys.stderr)
    _stderr_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    logger.addHandler(_stderr_handler)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)


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


# TASK-2197 (epic 2190 W4.1) — multiprocessor-mode vocabulary. Mode 2 = GPU
# offload; mode 1 = OpenMP (CPU). Named constants (were magic 1/2 literals).
_MULTIPROCESSOR_OPENMP = 1
_MULTIPROCESSOR_GPU = 2


def _resolve_multiprocessor_mode(input_data):
    """Resolve multiprocessor_mode: env OVERRIDES scenario.json (TASK-2197).

    ``RUN_ANUGA_MULTIPROCESSOR_MODE`` is how the dispatcher
    (``gn_anuga.services.resolve_target_dispatch``) drives mode=2 for the
    ``batch-gpu-a10g`` compute target — via a container-override env var, so
    the (immutable) scenario package never has to be mutated per-target.
    ``scenario_config['multiprocessor_mode']`` remains the fallback for an
    ad-hoc / non-dispatcher-driven run (e.g. a local GPU benchmark).

    A present-but-invalid env value logs a warning and falls through to the
    scenario_config source (same precedence as "env absent"); an absent/blank
    env is silently treated as absent. Both sources apply the TASK-1954-review
    falsy-mode->1 coercion (5d5328f): ``'0' or 1`` would otherwise stay the
    truthy string ``'0'`` and yield mode 0 — coercing to ``int`` FIRST then
    ``or 1`` fixes that.
    """
    env_raw = os.environ.get('RUN_ANUGA_MULTIPROCESSOR_MODE', '').strip()
    if env_raw:
        try:
            return int(env_raw) or _MULTIPROCESSOR_OPENMP
        except (TypeError, ValueError):
            logger.warning(
                "run_sim: invalid RUN_ANUGA_MULTIPROCESSOR_MODE=%r; "
                "falling back to scenario_config", env_raw,
            )
    try:
        return int(
            input_data['scenario_config'].get('multiprocessor_mode', _MULTIPROCESSOR_OPENMP)
        ) or _MULTIPROCESSOR_OPENMP
    except (TypeError, ValueError):
        logger.warning(
            "run_sim: invalid multiprocessor_mode in scenario_config; defaulting to 1"
        )
        return _MULTIPROCESSOR_OPENMP


def _get_module_offload_probe():
    """The process-wide offload probe. Upstream anuga develop exports
    ``gpu_offload_enabled`` at package level (anuga/__init__.py); the fork
    predates it. Returns the callable, or None on engines without it.
    Separate helper so tests can monkeypatch it hermetically."""
    try:
        from anuga import gpu_offload_enabled
    except Exception:
        return None
    return gpu_offload_enabled


def _assert_gpu_engaged(domain, requested_mode):
    """FAIL the run if mode 2 (GPU) was requested but did not actually engage
    (TASK-2197, epic 2190 W4.1; signal set corrected by the W4 adversarial
    review P0, verified against upstream anuga develop@57a64abf — the engine
    the GPU image actually bakes).

    Two DISTINCT fallbacks must both fail the run:

    * MODE fallback — the engine downgraded mode 2 to mode 1 ('legacy'):
      upstream does this under MPI without an MPI-enabled gpu_ext build; the
      fork does it (via a ``set_multiprocessor_mode`` exception that run_sim
      catches) when cupy/GPU is missing. Caught by the mode-retention check.
    * OFFLOAD fallback — upstream develop >= 9c409229 KEEPS
      ``multiprocessor_mode=2`` when 'unified' resolves to CPU-multicore
      (offload unsupported / disabled / no device): mode retention reads 2 on
      a run that never touches the GPU. Caught only by the offload signals:
      ``domain.gpu_offload_active`` (stamped at gpu-interface init from the
      process-wide state), a per-domain ``gpu_offload_enabled()`` method if
      an engine variant exposes one, or the package-level
      ``anuga.gpu_offload_enabled()`` (upstream exports it; the fork does
      not). The package probe is consulted only when the domain exposes no
      offload signal (e.g. upstream's CUDA/CuPy interface path never stamps
      ``gpu_offload_active``).

    Every signal the engine exposes must AGREE; any exposed signal reporting
    "not engaged" fails the run. An engine exposing NO signal at all fails
    closed. An engine exposing only the mode signal (the fork) passes on
    retention alone — pre-offload-selector engines have no silent CPU-mode-2:
    their mode 2 IS the GPU path, and their GPU-unavailable path drops the
    mode (caught above).

    No-op when ``requested_mode`` is not 2 — a CPU image has no GPU probe at
    all, so this must never be reached (let alone raise) on the CPU path.
    """
    if requested_mode != _MULTIPROCESSOR_GPU:
        return

    failures = []
    checked = []

    # Mode retention — catches both engines' mode-1 fallbacks.
    getter = getattr(domain, 'get_multiprocessor_mode', None)
    actual_mode = getter() if callable(getter) else getattr(domain, 'multiprocessor_mode', None)
    if actual_mode is not None:
        checked.append(f"multiprocessor_mode={actual_mode!r}")
        if actual_mode != _MULTIPROCESSOR_GPU:
            failures.append(
                f"the engine's active multiprocessor_mode is {actual_mode!r} "
                "— GPU offload did not engage (silent CPU fallback)"
            )

    # Per-domain resolved offload state (upstream: stamped at gpu-interface
    # init from the process-wide gpu_offload_enabled()).
    offload_checked = False
    active = getattr(domain, 'gpu_offload_active', None)
    if active is not None:
        offload_checked = True
        checked.append(f"gpu_offload_active={bool(active)!r}")
        if not active:
            failures.append(
                "domain.gpu_offload_active is False — 'unified' is running "
                "CPU-multicore with no GPU offload"
            )

    # Per-domain probe method, if an engine variant exposes one.
    probe = getattr(domain, 'gpu_offload_enabled', None)
    if callable(probe):
        offload_checked = True
        try:
            engaged = bool(probe())
        except Exception as exc:
            failures.append(f"gpu_offload_enabled() raised: {exc!r}")
        else:
            checked.append(f"domain.gpu_offload_enabled()={engaged!r}")
            if not engaged:
                failures.append("gpu_offload_enabled() reports False")

    # Package-level probe — only when the domain exposed no offload signal.
    if not offload_checked:
        module_probe = _get_module_offload_probe()
        if callable(module_probe):
            try:
                engaged = bool(module_probe())
            except Exception as exc:
                failures.append(f"anuga.gpu_offload_enabled() raised: {exc!r}")
            else:
                checked.append(f"anuga.gpu_offload_enabled()={engaged!r}")
                if not engaged:
                    failures.append(
                        "anuga.gpu_offload_enabled() reports False — "
                        "process-wide GPU offload is off (unsupported build, "
                        "no device, or disabled at launch)"
                    )

    if failures:
        raise RuntimeError(
            f"GPU offload requested (mode={_MULTIPROCESSOR_GPU}) but did not "
            f"engage: {'; '.join(failures)} — refusing to run silently in "
            "CPU mode."
        )

    if not checked:
        raise RuntimeError(
            f"GPU offload requested (mode={_MULTIPROCESSOR_GPU}) but the "
            "engine exposes no gpu_offload_enabled()/gpu_offload_active/"
            "multiprocessor_mode signal to verify it — refusing to run "
            "unverified."
        )

    logger.info("run_sim: GPU engagement verified via %s", ", ".join(checked))


def _capture_gpu_model():
    """Best-effort GPU model string (TASK-2197) — NVML (pynvml) first, then
    the ``nvidia-smi`` CLI. Import/getattr-defensive; NEVER raises. Runs on
    every mode-2 request, so a probe failure here must degrade to ``None``,
    not become a second failure mode alongside :func:`_assert_gpu_engaged`.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            return name.decode() if isinstance(name, bytes) else name
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            name = result.stdout.strip().splitlines()[0].strip()
            return name or None
    except Exception:
        pass
    return None


def run_sim(package_dir, username=None, password=None, batch_number=1, checkpoint_time=None, callback=None):
    # Lazy imports — these are only needed when actually running a simulation.
    anuga = import_optional("anuga")
    pickle = import_optional("dill")
    psutil = import_optional("psutil")
    from anuga import distribute, finalize, barrier, Inlet_operator
    from anuga.utilities import quantity_setting_functions as qs
    from anuga.operators.rate_operators import Polygonal_rate_operator

    # Clear any phase/mesh-feature state from a prior run on this process — a
    # localhost celery-anuga worker is long-lived and reused (TASK-1910).
    phase_tracker.reset()

    # Keep run_args for backward compat with update_web_interface in main() error handler.
    run_args = RunContext(package_dir, username, password)
    input_data = setup_input_data(package_dir)
    run_args.scenario_config = input_data['scenario_config']

    if callback is None and os.environ.get('HYDRATA_INTERNAL_COMPUTE_TOKEN'):
        callback = HydrataCallback.from_config(input_data['scenario_config'])
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
            # Sub-phase attribution (TASK-1910): tag the mesh-GENERATION window so
            # the resource sampler attributes its peak RSS to 'mesh-gen' (run 1260
            # / 8.16M tri OOM'd HERE). create_anuga_mesh returns the mesh; capture
            # the triangle count as a corpus feature joining peak memory to size.
            with phase_tracker.phase(phase_tracker.PHASE_MESH_GEN):
                if not os.path.isfile(input_data['mesh_filepath']):
                    _, anuga_mesh = create_anuga_mesh(input_data)
                    try:
                        # W3 (TASK-1923): route compute_mesh_qa shape + area metrics
                        # into the features bag so the corpus joins peak memory to
                        # mesh quality. compute_mesh_qa is numpy-only, no ANUGA import.
                        qa = compute_mesh_qa(anuga_mesh)
                        phase_tracker.set_mesh_features(
                            mesh_triangle_count=qa['triangle_count'],
                            mesh_node_count=qa['node_count'],
                            min_triangle_area=qa['min_triangle_area'],
                            area_histogram=qa['area_histogram'],
                            min_angle_deg=qa['min_angle_deg'],
                            sliver_count=qa['sliver_count'],
                            aspect_ratio_max=qa['aspect_ratio_max'],
                        )
                    except Exception:
                        logger.warning("could not record mesh-size features", exc_info=True)
                domain = anuga.Domain(
                    mesh_filename=input_data['mesh_filepath'],
                    use_cache=False,
                    verbose=False,
                )
            # Fallback feature source for the pre-existing-mesh path (a checkpoint
            # rebuild reuses the .msh, so create_anuga_mesh is skipped): read the
            # element count straight off the built Domain's mesh.
            try:
                if not phase_tracker.get_mesh_features().get("mesh_triangle_count"):
                    phase_tracker.set_mesh_features(
                        mesh_triangle_count=int(domain.mesh.number_of_triangles),
                    )
            except Exception:
                logger.debug("domain.mesh.number_of_triangles unavailable", exc_info=True)
            # PRE-FLIGHT (TASK-1138): surface nodata-under-mesh as a clear,
            # actionable error here rather than as an opaque exception deep
            # inside composite_quantity_setting_function. nan_treatment stays
            # 'exception' — we never silently fabricate bed elevation.
            # Sub-phase attribution (TASK-1910): the elevation raster read window.
            # assert_raster_has_no_nodata_inside_boundary + set_quantity both pull
            # the full DEM into memory (spatialInputUtil.py full-DEM read — a named
            # build-phase OOM consumer), so the sampler attributes its peak here.
            with phase_tracker.phase(phase_tracker.PHASE_RASTER_READ):
                assert_raster_has_no_nodata_inside_boundary(
                    input_data['elevation_filename'],
                    input_data['boundary_polygon'],
                    quantity_name='elevation',
                )
                poly_fun_pairs = [['Extent', input_data['elevation_filename']]]
                elevation_function = qs.composite_quantity_setting_function(
                    poly_fun_pairs,
                    domain,
                    nan_treatment='exception',
                )
                domain.set_quantity('elevation', elevation_function, verbose=False, alpha=0.99, location='centroids')

            # TASK-1299: post-mesh Raised structure elevation correction.
            # Apply per-structure height additions AFTER the base DEM is seated.
            # Only structures with method='Raised' are modified; Reflective and
            # Mannings are untouched (Reflective is a mesh void; Mannings is friction).
            # This replaces the old universal +5m gdal_rasterize burn (removed in 1270).
            raised_pairs = make_raised_elevation_pairs(input_data)
            if raised_pairs:
                logger.critical(f"Applying raised elevation for {len(raised_pairs)} Raised structure(s)")
                try:
                    # TASK-2149 F1: seat Raised heights via ABSOLUTE centroids so the
                    # point-in-polygon test matches the absolute-UTM Raised polygons
                    # regardless of the mesh geo_reference offset (previously absolute=False
                    # silently dropped every Raised structure on any local-offset mesh).
                    applied = apply_raised_elevation_correction(domain, raised_pairs)
                    logger.critical(f"Raised elevation applied for {applied}/{len(raised_pairs)} structure(s)")
                except Exception as e:
                    logger.error(f"Failed to apply raised elevation: {e} — continuing without Raised correction")

            if input_data['scenario_config'].get('store_mesh'):
                if getattr(domain, "dump_shapefile", None):
                    shapefile_name = f"{input_data['output_directory']}/{input_data['scenario_config'].get('run_id')}_{input_data['scenario_config'].get('id')}_{input_data['scenario_config'].get('project')}_mesh"
                    logger.info(f"mesh shapefile: {shapefile_name}")
                    domain.dump_shapefile(
                        shapefile_name=shapefile_name,
                        epsg_code=input_data['scenario_config'].get('epsg')
                    )
            frictions = make_frictions(input_data)
            # PRE-FLIGHT (TASK-1138): only a friction RASTER is nodata-checkable.
            # make_frictions (TASK-1259) returns a list that merges the optional
            # ['Extent', raster] pair with any per-structure Manning's-n polygon
            # patches; the next() below extracts the raster pair (if present) to
            # nodata-check it. composite_quantity_setting_function below uses
            # anuga's default nan_treatment='exception'.
            friction_raster_pair = next(
                (pair for pair in frictions
                 if len(pair) == 2 and pair[0] == 'Extent'),
                None,
            )
            # Sub-phase attribution (TASK-1910): the friction raster read window
            # (same full-raster pull as elevation). Disjoint from the elevation
            # window above — the sampler max-accumulates both into 'raster-read'.
            with phase_tracker.phase(phase_tracker.PHASE_RASTER_READ):
                if friction_raster_pair is not None:
                    assert_raster_has_no_nodata_inside_boundary(
                        friction_raster_pair[1],
                        input_data['boundary_polygon'],
                        quantity_name='friction',
                    )
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
            # Sub-phase attribution (TASK-1910): distribute() partitions the mesh
            # on rank 0 then scatters sub-domains to the worker ranks — the named
            # build-phase OOM consumer that killed run 1253 (4.35M tri). Tagged as
            # 'distribute' (the partition step lives inside distribute() in
            # anuga_core, which is out of scope to split finer here).
            with phase_tracker.phase(phase_tracker.PHASE_DISTRIBUTE):
                domain = distribute(domain, verbose=True)
                # Skip flux reconstruction for dry cells — free speedup, no
                # accuracy impact (ported from run_anuga main, dfc6fa4/58c5604).
                domain.optimise_dry_cells = True

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

        # TASK-1954 (epic 1952): GPU-mode flag — read multiprocessor_mode from
        # scenario.json (default 1 = OpenMP, preserves today's behaviour exactly).
        # Called AFTER set_boundary (ordering constraint: boundaries must be set
        # before the solver mode is configured) and BEFORE evolve.
        # Mode 2 = GPU offload via NVIDIA HPC SDK (nvc -mp=gpu); mode 1 = OpenMP.
        # ScenarioConfig allows extra fields (model_config extra=allow) so
        # multiprocessor_mode passes through unvalidated when absent.
        # TASK-2197 (epic 2190 W4.1): env now beats scenario.json (see
        # _resolve_multiprocessor_mode) — the package stays immutable.
        _multiprocessor_mode = _resolve_multiprocessor_mode(input_data)
        try:
            domain.set_multiprocessor_mode(_multiprocessor_mode)
            logger.info("run_sim: multiprocessor_mode=%s", _multiprocessor_mode)
        except Exception:
            logger.warning(
                "run_sim: set_multiprocessor_mode(%s) failed; continuing",
                _multiprocessor_mode, exc_info=True,
            )
        # TASK-2197 — a mode-2 request that did not actually engage GPU
        # offload now FAILS the run (was silent). No-op for mode 1.
        _assert_gpu_engaged(domain, _multiprocessor_mode)
        # TASK-2197 — capture the actual mode + (best-effort) GPU model into
        # the resource-summary feature bag alongside mesh_triangle_count etc,
        # so BatchJobResourceRecord carries per-run hardware proof.
        # AdminRunResourceRecordSerializer.get_mode/get_gpu_model already read
        # raw['features'] (TASK-2195) — this is the write side. gpu_model is
        # omitted entirely (never a fabricated None) on the CPU path.
        _gpu_features = {
            'mode': 'gpu' if _multiprocessor_mode == _MULTIPROCESSOR_GPU else 'cpu',
        }
        if _multiprocessor_mode == _MULTIPROCESSOR_GPU:
            _gpu_model = _capture_gpu_model()
            if _gpu_model:
                _gpu_features['gpu_model'] = _gpu_model
            # Per-domain resolved offload state (upstream engines stamp it at
            # gpu-interface init) — the carrier's hardware proof reads this
            # alongside gpu_model. bool() cast: never a numpy/engine type in
            # the summary JSON (the 2033 telemetry lesson). Omitted (not
            # fabricated) when the engine doesn't expose it.
            _offload_active = getattr(domain, 'gpu_offload_active', None)
            if _offload_active is not None:
                _gpu_features['gpu_offload_active'] = bool(_offload_active)
        phase_tracker.set_mesh_features(**_gpu_features)

        # W3 (TASK-1923): record BC types + scenario denorms AFTER set_boundary
        # so domain.boundary reflects the actual tags used for this run.
        # W3 (TASK-1927): also stamp experiment_tag from ANUGA_EXPERIMENT_TAG env
        # (injected by _dispatch_batch when run.experiment_tag is set). Absent for
        # ad-hoc runs; the corpus export will see None for those rows.
        try:
            bc_types = extract_boundary_condition_types(domain)
            sc = input_data.get('scenario_config') or {}
            _exp_tag = os.environ.get('ANUGA_EXPERIMENT_TAG') or None
            phase_tracker.set_mesh_features(
                boundary_condition_types=bc_types,
                resolution=sc.get('resolution'),
                duration=duration,
                experiment_tag=_exp_tag,
            )
        except Exception:
            logger.warning("could not record BC/scenario features", exc_info=True)

        # W3 (TASK-1924): emit an early PARTIAL resource_summary now that all
        # mesh features are stamped but before the long evolve loop begins.
        # An evolve crash (OOM mid-run) will still leave a ledger row with mesh
        # features. The final run-end report_resource_summary supersedes this.
        try:
            callback.on_mesh_features_ready()
        except Exception:
            logger.warning("on_mesh_features_ready failed; suppressed", exc_info=True)

        yieldstep = compute_yieldstep(duration)
        checkpoint_directory = input_data['checkpoint_directory']
        # W2 (TASK-1919) — Disable checkpoint WRITING on the Batch/no-resume path.
        # TASK-1048 confirmed there is no checkpoint-resume on AWS Batch (spot loss
        # accepted); the pickles are pure scratch, filling the root volume linearly
        # at checkpoint_step=1 (one full-domain pickle per rank per yieldstep).
        # On Batch (AWS_BATCH_JOB_ID present) disable writing entirely.
        # Explicit override env RUN_ANUGA_CHECKPOINTS={"on","off"} lets operators
        # force either way and makes the behaviour unit-testable.
        _ckpt_env = os.environ.get("RUN_ANUGA_CHECKPOINTS", "").strip().lower()
        _on_batch = bool(os.environ.get("AWS_BATCH_JOB_ID"))
        if _ckpt_env == "on":
            _enable_checkpoints = True
        elif _ckpt_env == "off":
            _enable_checkpoints = False
        else:
            # Default: OFF on Batch, ON locally.
            _enable_checkpoints = not _on_batch
        if _enable_checkpoints:
            logger.info("run_sim: checkpointing ENABLED (checkpoint_dir=%s)", checkpoint_directory)
        else:
            logger.info(
                "run_sim: checkpointing DISABLED on %s path (RUN_ANUGA_CHECKPOINTS=%r)",
                "Batch" if _on_batch else "forced-off",
                _ckpt_env or "(default)",
            )
        domain.set_checkpointing(
            checkpoint=_enable_checkpoints,
            checkpoint_dir=checkpoint_directory,
            checkpoint_step=1,
        )
        barrier()
        # Per-yieldstep diagnostics + run summary (ported from run_anuga main,
        # 58c5604). Rank 0 owns the monitor: it computes stability/flow metrics
        # from its sub-domain and writes run_diagnostics_N.csv + run_summary_N.json.
        # The run_summary's outcome/stability fields are the Merewether e2e's
        # correctness backstop. Domain time is relative (0…duration) on this
        # branch (no set_starttime), so record() takes t directly.
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
        # W6 (TASK-1044) — `simulation_start` is the absolute wall-clock anchor used
        # for ETA estimation; `start` is the per-tick reference reset every iteration.
        simulation_start = time.time()
        start = simulation_start
        # Sub-phase attribution (TASK-1910): the timestepping solver loop. Set
        # (not context-managed) because the loop is the last build phase before
        # post-processing; the phase is cleared after the trailing barrier below.
        phase_tracker.set_phase(phase_tracker.PHASE_EVOLVE)
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
                mem_mb = memory_usage / (1024 * 1024)
                # Per-yieldstep diagnostics. t is already simulation-relative here
                # (finaltime=duration, no set_starttime), so pass it straight through.
                diag = monitor.record(t, wall_time_s=stop - start, mem_mb=mem_mb)
                logger.info(
                    f'{percentage_done}% | {minutes}m {seconds}s | '
                    f'mem: {memory_percent}% | disk: {psutil.disk_usage("/").percent}% | '
                    f'{domain.get_datetime().isoformat()} | '
                    + monitor.format_log_suffix(diag)
                )
                start = time.time()
        # Evolve done — clear the phase so post-processing (sww_merge / result
        # publish) isn't attributed to 'evolve' (TASK-1910).
        phase_tracker.set_phase(None)
        barrier()
        domain.sww_merge(verbose=True, delete_old=True)
        barrier()
        if anuga.myid == 0:
            # Write run_diagnostics_N.csv + run_summary_N.json (stability/outcome).
            # Guarded: telemetry failure must not abort the handoff (TASK-2033).
            finalize_monitor_safely(monitor)
            max_memory_usage = int(round(max(memory_usage_logs)))
            callback.on_metric('memory_used', max_memory_usage)
            logger.info("Processing results...")
            # TASK-1954: tag post_process_sww as the 'cog-export' phase so
            # observed.phase_durations_s captures the publish taxonomy
            # (build = mesh-gen + raster-read + distribute, solve = evolve,
            # publish = cog-export + archive).
            with phase_tracker.phase(phase_tracker.PHASE_COG_EXPORT):
                post_process_sww(package_dir, run_args=run_args)
    except Exception:
        callback.on_status('error')
        logger.error(f"{traceback.format_exc()}")
        # Belt-and-braces in case a caller clobbered logger.handlers.
        print(traceback.format_exc(), file=sys.stderr)
        sys.stderr.flush()
        # Tear down COMM_WORLD so other ranks don't spin at the next barrier.
        try:
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
        except Exception:
            logger.exception('MPI_Abort failed')
        raise
    finally:
        try:
            _finalize_with_timeout(finalize)
        finally:
            try:
                callback.close()
            except Exception:
                logger.exception('callback.close() raised; suppressed')
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
        # Prefer Batch token-auth (RAW X-Internal-Token header, not Bearer);
        # fall back to BasicAuth for localhost/legacy.
        token = os.environ.get('HYDRATA_INTERNAL_COMPUTE_TOKEN')
        if not token and not (username and password):
            return
        if run_args.scenario_config is not None:
            scenario_config = run_args.scenario_config
        else:
            # run_sim failed before setup_input_data populated the cache.
            scenario_json_path = os.path.join(package_dir, 'scenario.json')
            with open(scenario_json_path, 'r') as f:
                scenario_config = json.load(f)
        run_id = scenario_config.get('run_id')
        control_server = scenario_config.get('control_server')
        if not (control_server and run_id):
            return
        from run_anuga._http import make_internal_session, post_to_control_server

        url = f"{control_server}api/v2/anuga/runs/{run_id}/error/"
        # Small POST with a scalar message: a 30s upper bound is fine here
        # (the helper default is None / no timeout, which is required for the
        # PATCH-with-files callers but inappropriate for an error report).
        if token:
            session = make_internal_session(token)
            try:
                post_to_control_server(
                    url,
                    method="POST",
                    data={'message': message},
                    session=session,
                    timeout=30,
                )
            finally:
                session.close()
        else:
            requests = import_optional("requests")
            auth = requests.auth.HTTPBasicAuth(username, password)
            post_to_control_server(
                url, auth=auth, method="POST", data={'message': message}, timeout=30,
            )
    except Exception:
        logger.exception("Failed to report run error to control server")


if __name__ == '__main__':
    main()
