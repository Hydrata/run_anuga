"""Result-handoff stage for ANUGA runs.

After ``run_sim`` finishes, the result outputs need to be:

1. Zipped into a single archive (mirroring ``batch/entrypoint.sh`` lines 102-109).
2. Uploaded to an S3 result bucket.
3. Announced to the Hydrata control server via ``POST /api/v2/anuga/runs/<id>/process-result/``
   so the BE can dispatch the ``process_result_async`` celery task.

On a hard failure, the run must be reported via ``POST /error/`` so it does
not wedge in ``COMPUTING`` until the 1h zombie watchdog flips it to ERROR
(the surface that TASK-1158 just paid for under canary-19).

Pre-F1 this lived in two places:

* ``batch/entrypoint.sh`` — shell + ``aws s3 cp`` + ``curl`` (Batch path).
* ``compute_anuga.send_process_result`` — Django/Python helper (localhost path).

F1 (TASK-1159) consolidates both into this module so:

* The Batch entrypoint shrinks to "download + invoke run_anuga".
* F2 (TASK-1161) ``_dispatch_local`` rewrite reuses ``run_and_report`` instead
  of growing its own copy.
* The wire field name lives in ONE constant (``RESULT_PACKAGE_KEY_FIELD``)
  that the Django receiver imports too, making the F0 drift class
  structurally impossible.

The module is Django-free; ``requests`` and ``boto3`` are pulled lazily via
``import_optional`` so pure-disk CLI/OSS users do not need the
``[platform]`` extra to import this file.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
import zipfile
from pathlib import Path
from typing import Any

from run_anuga._imports import import_optional
from run_anuga._logging import install_mname_filter

logger = logging.getLogger(__name__)
install_mname_filter(logger)


# Single source of truth for the wire field name. Both the Python sender below
# and the Django receiver (``gn_anuga.api_v2.HydrataAnugaRunsViewSet.process_result``)
# read from this constant — the F0 drift class (``key`` vs ``result_package_key``)
# cannot recur as long as both sides import it.
RESULT_PACKAGE_KEY_FIELD = "result_package_key"


def make_result_key(project_id: int, scenario_id: int, run_id: int) -> str:
    """Return the canonical S3 key for a run's result zip.

    Mirrors ``batch/entrypoint.sh`` line 34 (``${PROJECT_ID}_${SCENARIO_ID}_${RUN_ID}_results.zip``).
    """
    return f"{project_id}_{scenario_id}_{run_id}_results.zip"


# Directory names whose entire subtree is excluded from the result package
# (matched on any path component, at any depth). See ``_is_excluded_from_result``.
_EXCLUDED_DIR_NAMES = frozenset({"run_anuga", "checkpoints", "videos"})


def _is_excluded_from_result(relative: Path, result_zip_name: str) -> bool:
    """Return True if ``relative`` (a path under the package dir) must NOT be
    written into the result package.

    Result-package slimming (TASK-1821). The only consumer of the uploaded
    result zip is the Hydrata backend's ``Run.process_result``
    (gn_anuga/models/run.py), which extracts ONLY the ``*_max.tif`` rasters
    (depth / velocity / depthIntegratedVelocity). Everything else the runner
    leaves in the output directory is, for the result zip's purpose, dead
    weight — and at production scale the raw ``.sww`` (tens to >100 GB) made the
    package overflow the Batch host disk AND dominated the zip+upload wall time
    (the handoff was output-bound, not compute-bound). See
    docs/reports/2026-06-19-anuga-x32-w0-benchmark.md and TASK-1820/1821.

    Excluded:
      ``*.sww``        Raw ANUGA NetCDF. Fully consumed on-box by
                       ``post_process_sww`` (-> the max + per-timestep TIFs)
                       BEFORE this zip is built; no BE/FE path reads it from the
                       zip and no reprocess-from-sww code path exists. The
                       dominant bulk at production scale.
      ``checkpoints/`` MPI per-rank checkpoint pickles. D4.c (TASK-1048): no
                       checkpoint resume — operator accepts spot loss. Pure
                       scratch; scales with rank count (P32 -> 32 pickles per
                       checkpoint time) and dominates small/short packages.
      ``*_Time_*.tif`` Per-timestep rasters from ``Make_Geotif(myTimeStep='all')``.
                       ``process_result`` reads only ``*_max.tif``; per-timestep
                       rasters are delivered (when at all) by the separate
                       ``generate_stac`` upload to ``ANUGA_S3_STAC_BUCKET_NAME``,
                       which is NOT invoked on the Batch ``run-and-report`` path
                       — so they are unread dead weight in the result zip.
      ``*.msh``        The mesh is persisted independently via
                       ``Run.msh_snapshot`` (AnugaMeshStorage); this copy is
                       redundant.
      ``videos/``      Already removed by ``post_process_sww`` before handoff;
                       listed here for defence in depth.

    Kept: the ``*_max.tif`` rasters (the BE payload), run logs, ``scenario.json``,
    and the small ``inputs/`` tree (bounded provenance). Plus the always-excluded
    ``package.zip`` (the downloaded input) and the result zip itself.
    """
    if any(part in _EXCLUDED_DIR_NAMES for part in relative.parts[:-1]):
        return True
    name = relative.name
    if name in ("package.zip", result_zip_name):
        return True
    if name.endswith((".sww", ".msh")):
        return True
    if name.endswith(".tif") and "_Time_" in name:
        return True
    return False


def zip_outputs(package_dir: str | Path, result_zip_path: str | Path) -> Path:
    """Zip the slimmed result payload from ``package_dir`` into ``result_zip_path``.

    Writes only the artifacts the Hydrata backend consumes plus small
    provenance (max-quantity rasters, run logs, ``scenario.json``, ``inputs/``);
    the multi-GB raw ``.sww``, MPI ``checkpoints/``, per-timestep ``*_Time_*.tif``
    rasters, redundant ``.msh`` mesh, ``package.zip``, the result zip itself, and
    any embedded ``run_anuga/`` source tree are excluded. The exclusion rules
    (and why each is safe) live in ``_is_excluded_from_result``; this Python is
    the single source of truth for the result-zip contents (it supersedes the
    historical ``zip -x`` line in ``batch/entrypoint.sh``).
    """
    package_dir = Path(package_dir).resolve()
    result_zip_path = Path(result_zip_path).resolve()
    result_zip_name = result_zip_path.name

    with zipfile.ZipFile(result_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in package_dir.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(package_dir)
            if _is_excluded_from_result(relative, result_zip_name):
                continue
            zf.write(path, arcname=str(relative))

    return result_zip_path


def upload_result_to_s3(
    zip_path: str | Path,
    bucket: str,
    key: str,
) -> None:
    """Upload ``zip_path`` to ``s3://<bucket>/<key>`` via boto3.

    ``boto3`` is pulled lazily so this module remains importable on pure-disk
    OSS installs that did not pull the ``[platform]`` extra.

    Credentials come from the standard boto3 chain (env vars, instance
    profile, ``~/.aws/credentials``); the function deliberately does NOT
    accept explicit keys — Batch uses the task role and localhost uses the
    site IAM user (see ``rules/credentials.md``).
    """
    boto3 = import_optional("boto3")
    s3 = boto3.client("s3")
    s3.upload_file(str(zip_path), bucket, key)


def report_result(
    control_server: str,
    run_id: int,
    token: str,
    result_key: str,
    *,
    session: Any = None,
    timeout: int = 30,
) -> Any:
    """POST ``{result_package_key: result_key}`` to ``/api/v2/anuga/runs/<run_id>/process-result/``.

    Returns the ``requests.Response`` so callers can inspect the status code.
    On non-2xx the helper logs (does not raise); callers MUST inspect the
    response and surface a failure to keep the run from wedging in COMPUTING.
    """
    from run_anuga._http import make_internal_session, post_to_control_server

    url = f"{control_server.rstrip('/')}/api/v2/anuga/runs/{run_id}/process-result/"
    owns_session = session is None
    if owns_session:
        session = make_internal_session(token)
    try:
        return post_to_control_server(
            url,
            method="POST",
            data={RESULT_PACKAGE_KEY_FIELD: result_key},
            session=session,
            timeout=timeout,
        )
    finally:
        if owns_session:
            session.close()


def report_error(
    control_server: str,
    run_id: int,
    token: str,
    message: str,
    *,
    source: str | None = None,
    session: Any = None,
    timeout: int = 30,
) -> Any:
    """POST ``{message, source?}`` to ``/api/v2/anuga/runs/<run_id>/error/``.

    Mirrors the entrypoint EXIT trap (``batch/entrypoint.sh`` lines 44-51) so
    a failed run flips to ERROR instead of wedging in COMPUTING. ``source``
    is an optional free-text tag (``"entrypoint.sh"``, ``"run_and_report"``,
    etc.) the BE writes verbatim into the run log.
    """
    from run_anuga._http import make_internal_session, post_to_control_server

    url = f"{control_server.rstrip('/')}/api/v2/anuga/runs/{run_id}/error/"
    payload: dict[str, Any] = {"message": message}
    if source:
        payload["source"] = source
    owns_session = session is None
    if owns_session:
        session = make_internal_session(token)
    try:
        return post_to_control_server(
            url,
            method="POST",
            data=payload,
            session=session,
            timeout=timeout,
        )
    finally:
        if owns_session:
            session.close()


def _read_scenario_config(package_dir: Path) -> dict:
    """Read ``scenario.json`` from a package directory.

    Returns the parsed dict. Raises ``FileNotFoundError`` if absent (the
    caller must surface this via /error/ so the run doesn't wedge).
    """
    scenario_path = package_dir / "scenario.json"
    with scenario_path.open() as fp:
        return json.load(fp)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} env var is required")
    return value


def _is_mpi_rank_zero() -> bool:
    """Return True on the only rank that should run the handoff stages.

    The handoff (zip + upload + POST) must run exactly once. When run under
    ``mpirun -np N`` all N ranks reach this code; only rank 0 does the I/O.
    When no MPI is loaded (a localhost CLI run), there is one process and it
    IS rank 0.
    """
    try:
        from mpi4py import MPI

        # anuga's run_sim finalizes MPI internally; calling Get_rank() after
        # MPI_FINALIZE is illegal and aborts the process. In the single-process
        # CLI case mpi4py auto-inits on import but anuga then finalizes, so
        # post-sim callers see Is_finalized()=True. A single-process run is
        # always rank 0.
        if MPI.Is_finalized():
            return True
        return MPI.COMM_WORLD.Get_rank() == 0
    except ImportError:
        return True


def run_and_report(
    package_dir: str | Path,
    *,
    callback: Any = None,
    result_bucket: str | None = None,
) -> dict:
    """Run an ANUGA simulation, zip+upload the results, and POST /process-result/.

    Single entry point that both the Batch entrypoint and the F2 localhost
    dispatcher invoke. On any failure, POSTs /error/ before re-raising so the
    BE-side run row flips to ERROR instead of wedging in COMPUTING.

    Parameters
    ----------
    package_dir
        Path to the unzipped scenario package (contains scenario.json + inputs/).
    callback
        Optional ``SimulationCallback``. When ``None`` and the token env var
        is set, ``run_sim`` auto-constructs a ``HydrataCallback``; when neither
        the callback nor the token is present, ``run_sim`` falls back to
        ``NullCallback``. Pass ``LoggingCallback()`` for a silent stdout-only run.
    result_bucket
        S3 bucket for the result zip. Defaults to the ``RESULT_S3_BUCKET`` env
        var (matches ``batch/entrypoint.sh`` line 9).

    Returns
    -------
    dict
        ``{"result_key": <s3 key>, "process_result_status": <int>}`` on success.

    Raises
    ------
    Any exception from ``run_sim`` (after /error/ is POSTed).
    """
    package_dir = Path(package_dir).resolve()

    # All ranks need the simulation. Only rank 0 does the post-sim handoff.
    from run_anuga.run import run_sim

    scenario_config = _read_scenario_config(package_dir)
    run_id = scenario_config.get("run_id")
    project_id = scenario_config.get("project")
    scenario_id = scenario_config.get("id")
    control_server = scenario_config.get("control_server")
    token = _required_env("HYDRATA_INTERNAL_COMPUTE_TOKEN")
    # Fail fast on the bucket too so a misconfigured worker doesn't burn N
    # hours of ANUGA compute before discovering it can't upload the result.
    bucket = result_bucket or _required_env("RESULT_S3_BUCKET")

    if not (run_id and project_id and scenario_id and control_server):
        raise RuntimeError(
            "scenario.json is missing one of run_id/project/id/control_server "
            f"(got run_id={run_id!r}, project={project_id!r}, id={scenario_id!r}, "
            f"control_server={control_server!r})"
        )

    try:
        run_sim(str(package_dir), callback=callback)
    except Exception as exc:
        if _is_mpi_rank_zero():
            try:
                report_error(
                    control_server,
                    run_id,
                    token,
                    message=f"{exc}\n{traceback.format_exc()}",
                    source="run_and_report",
                )
            except Exception:
                logger.exception("run_and_report: /error/ POST failed; suppressed")
        raise

    if not _is_mpi_rank_zero():
        return {"result_key": None, "process_result_status": None}

    result_key = make_result_key(project_id, scenario_id, run_id)
    result_zip_path = package_dir / result_key

    try:
        zip_outputs(package_dir, result_zip_path)
        upload_result_to_s3(result_zip_path, bucket, result_key)
        response = report_result(control_server, run_id, token, result_key)
    except Exception as exc:
        try:
            report_error(
                control_server,
                run_id,
                token,
                message=f"run_and_report handoff failed: {exc}",
                source="run_and_report",
            )
        except Exception:
            logger.exception("run_and_report: /error/ POST failed; suppressed")
        raise

    status_code = getattr(response, "status_code", None)
    if status_code is None or status_code >= 400:
        # Truncate the response body so a Django debug-HTML 500 doesn't bloat /error/.
        body = (getattr(response, "text", "") or "")[:500]
        message = f"/process-result/ returned HTTP {status_code}; body={body!r}"
        try:
            report_error(control_server, run_id, token, message=message, source="run_and_report")
        except Exception:
            logger.exception("run_and_report: /error/ POST failed; suppressed")
        raise RuntimeError(message)

    logger.info("run_and_report: /process-result/ returned %s for run %s", status_code, run_id)
    return {"result_key": result_key, "process_result_status": status_code}
