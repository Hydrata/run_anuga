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


def zip_outputs(package_dir: str | Path, result_zip_path: str | Path) -> Path:
    """Zip the package directory into ``result_zip_path``.

    Mirrors ``batch/entrypoint.sh`` lines 102-105:

    ``zip -q -r "${RESULT_ZIP}" . -x "package.zip" "run_anuga/*" "${RESULT_KEY}"``

    Excludes ``package.zip`` (the input the entrypoint downloaded), the result
    zip itself (so the output never includes itself recursively), and any
    ``run_anuga/`` source tree that may have been mounted into the working
    directory during testing.
    """
    package_dir = Path(package_dir).resolve()
    result_zip_path = Path(result_zip_path).resolve()
    result_zip_name = result_zip_path.name

    with zipfile.ZipFile(result_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in package_dir.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(package_dir)
            top = relative.parts[0]
            if top == "run_anuga" or relative.name == "package.zip" or relative.name == result_zip_name:
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
