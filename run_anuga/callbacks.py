"""
Simulation callback protocol and implementations.

Callbacks provide structured progress reporting for ``run_sim()``.
They replace the scattered ``update_web_interface()`` calls with a
clean, swappable interface:

* **NullCallback** — does nothing (default for standalone use).
* **LoggingCallback** — logs progress via Python logging (CLI mode).
* **HydrataCallback** — HTTP POST to the Hydrata control server V2 API
  using the ``X-Internal-Token`` header and an owned ``requests.Session``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Protocol, runtime_checkable

from run_anuga._logging import install_mname_filter

logger = logging.getLogger(__name__)
install_mname_filter(logger)  # TASK-1276: stamp mname/lnum for anuga's root formatter


@runtime_checkable
class SimulationCallback(Protocol):
    """Protocol for simulation progress reporting."""

    def on_status(self, status: str, **kwargs: Any) -> None:
        """Called when the simulation status changes (e.g. 'building mesh', '45.2%', 'error')."""
        ...

    def on_metric(self, key: str, value: Any) -> None:
        """Called to report a numeric metric (e.g. mesh_triangle_count, memory_used)."""
        ...

    def on_file(self, key: str, filepath: str) -> None:
        """Called to report an output file (e.g. video, raster)."""
        ...

    def on_progress(self, pct: float, eta_seconds: int | None = None) -> None:
        """Report scalar progress (numeric percentage + optional ETA in seconds).

        W6 (TASK-1044): replaces the legacy ``on_status('X%')`` overloading.
        ``on_status`` is now reserved for state-word transitions; numeric
        progress flows via this method.
        """
        ...

    def on_mesh_features_ready(self) -> None:
        """Called after mesh-gen + feature stamping, before the evolve loop (TASK-1924).

        Used to emit an early PARTIAL resource_summary so an evolve crash still
        leaves a ledger row with mesh features. Default is a no-op.
        """
        ...


class NullCallback:
    """Callback that silently discards all events.  Default for standalone use."""

    def on_status(self, status: str, **kwargs: Any) -> None:
        pass

    def on_metric(self, key: str, value: Any) -> None:
        pass

    def on_file(self, key: str, filepath: str) -> None:
        pass

    def on_progress(self, pct: float, eta_seconds: int | None = None) -> None:
        pass

    def on_mesh_features_ready(self) -> None:
        pass

    def close(self) -> None:
        """No-op so ``run_sim`` can always call ``callback.close()`` in ``finally``."""
        pass


class LoggingCallback:
    """Callback that logs events via Python logging.  Useful for CLI runs."""

    def __init__(self, logger_instance: logging.Logger | None = None):
        self._logger = logger_instance or logger

    def on_status(self, status: str, **kwargs: Any) -> None:
        self._logger.info("status: %s %s", status, kwargs if kwargs else "")

    def on_metric(self, key: str, value: Any) -> None:
        self._logger.info("metric: %s = %s", key, value)

    def on_file(self, key: str, filepath: str) -> None:
        self._logger.info("file: %s -> %s", key, filepath)

    def on_progress(self, pct: float, eta_seconds: int | None = None) -> None:
        self._logger.info('progress: %.1f%% eta=%ss', pct, eta_seconds)

    def on_mesh_features_ready(self) -> None:
        self._logger.info('mesh features ready (pre-evolve)')

    def close(self) -> None:
        """No-op so ``run_sim`` can always call ``callback.close()`` in ``finally``."""
        pass


class HydrataCallback:
    """Callback that reports progress to the Hydrata control-server V2 API.

    TASK-1049 (W1 of TASK-1048): rewritten around the V2 endpoints and the
    site-wide ``HYDRATA_INTERNAL_COMPUTE_TOKEN`` shared secret. The legacy
    BasicAuth path (``ANUGA_USERNAME`` / ``ANUGA_PASSWORD``) and the V1 URL
    template (``/anuga/api/{p}/{s}/run/{r}/``) are gone — this callback now
    talks to ``/api/v2/anuga/runs/{run_id}/log/`` and
    ``/api/v2/anuga/runs/{run_id}/progress/`` only. ``project_id`` and
    ``scenario_id`` are inferred server-side from the run row.

    Mirrors ``compute_anuga.tasks._SessionHTTPHandler`` (the canonical owned
    Session pattern shipped under TASK-961 / TASK-948 W5.1):

    * A single ``requests.Session`` is created in ``__init__`` and reused
      across every POST.
    * The ``X-Internal-Token`` header is pre-set on the session (RAW token,
      no ``Bearer`` prefix — see ``gn_anuga.permissions.IsInternalComputeCaller``).
    * ``close()`` releases the session's connection pool. It is idempotent
      (safe to call twice). Callers (typically ``run_sim``) MUST pair
      construction with ``try/finally: callback.close()``.

    Fail-fast: ``__init__`` raises ``RuntimeError`` if the token env var
    is missing or empty. This prevents silent 401 storms mid-run when the
    worker is mis-configured.

    Parameters
    ----------
    control_server : str
        Base URL, e.g. ``"https://hydrata.com/"`` (trailing slash optional).
    project : int
        Project ID. Retained for logging/back-compat with ``from_config``;
        NOT used in the V2 URL (server infers it from the run row).
    scenario : int
        Scenario ID. Retained for logging/back-compat; NOT in the V2 URL.
    run_id : int
        Run ID. This is the only ID that appears in the V2 URL.

    Raises
    ------
    RuntimeError
        If ``HYDRATA_INTERNAL_COMPUTE_TOKEN`` is unset or empty.
    """

    _TOKEN_ENV = 'HYDRATA_INTERNAL_COMPUTE_TOKEN'

    # W3 (TASK-1926) — heartbeat interval: POST a log line every N seconds during
    # on_progress calls even when no metric/status change occurs. A 21h run with
    # no status changes would otherwise leave no recent DB log before an OOM kill.
    # Default = 15 minutes; overridable via RUN_ANUGA_HEARTBEAT_INTERVAL_S env.
    HEARTBEAT_INTERVAL_S: float = float(
        os.environ.get('RUN_ANUGA_HEARTBEAT_INTERVAL_S', '900')  # 15 min
    )

    def __init__(
        self,
        control_server: str,
        project: int,
        scenario: int,
        run_id: int,
    ):
        from run_anuga._http import make_internal_session

        token = os.environ.get(self._TOKEN_ENV, '')
        if not token:
            raise RuntimeError(
                f"{self._TOKEN_ENV} env var is required for HydrataCallback"
            )

        self.control_server = control_server
        self.project = project
        self.scenario = scenario
        self.run_id = run_id
        self.session = make_internal_session(token)
        # W3 (TASK-1926): track last heartbeat time for periodic log-flush.
        self._last_heartbeat_t: float = time.time()

    @property
    def _log_url(self) -> str:
        """V2 log endpoint (TASK-987 — `/api/v2/anuga/runs/<id>/log/`)."""
        base = self.control_server.rstrip('/')
        return f"{base}/api/v2/anuga/runs/{self.run_id}/log/"

    @property
    def _progress_url(self) -> str:
        """V2 progress endpoint (TASK-995 — `/api/v2/anuga/runs/<id>/progress/`)."""
        base = self.control_server.rstrip('/')
        return f"{base}/api/v2/anuga/runs/{self.run_id}/progress/"

    def close(self) -> None:
        """Release the owned ``requests.Session`` connection pool.

        Idempotent: safe to call twice. ``run_sim`` pairs this with a
        ``try/finally`` block at the construction site (TASK-1049 W4
        of TASK-948 — owned Session lifecycle).
        """
        try:
            self.session.close()
        except Exception:  # pragma: no cover — defensive
            logger.debug('HydrataCallback.close() suppressed exception', exc_info=True)

    def _post(self, url: str, data: dict) -> None:
        """POST ``data`` (form-encoded) to ``url`` via the owned session.

        Auth flows via the pre-set ``X-Internal-Token`` header on the
        session; no ``auth=`` kwarg is passed. Network/transport failures
        are swallowed and logged — callbacks must never break the run loop.
        """
        from run_anuga._http import post_to_control_server

        try:
            post_to_control_server(
                url,
                method='POST',
                data=data,
                session=self.session,
                timeout=30,
            )
        except Exception:
            logger.exception('HydrataCallback POST to %s failed', url)

    def on_status(self, status: str, **kwargs: Any) -> None:
        """Report a state-word transition.

        Folded onto the V2 log endpoint as a tagged log line — the V2
        endpoint accepts ``{message, levelname, created}`` and writes the
        formatted line to ``Run.log``. No separate ``status`` field is
        sent (terminal state transitions are owned by the orchestrator,
        not the runner).
        """
        self._post(self._log_url, {
            'message': f"status: {status}",
            'levelname': 'INFO',
            'created': time.time(),
        })

    def on_metric(self, key: str, value: Any) -> None:
        """Report a numeric metric as a log line on the V2 log endpoint.

        TASK-1049 narrows the V1 PATCH-arbitrary-field surface: metrics
        now flow through the log channel as structured strings. Server-side
        metric persistence (if any) is the orchestrator's job.
        """
        self._post(self._log_url, {
            'message': f"metric: {key}={value}",
            'levelname': 'INFO',
            'created': time.time(),
        })

    def on_file(self, key: str, filepath: str) -> None:
        """Report an output file as a log line on the V2 log endpoint.

        TASK-1049: the V2 log endpoint does not accept file uploads — file
        artifact transport is handled separately by ``process_result_async``
        (the package zip uploaded to S3 from compute_anuga). This callback
        now just emits a log line so the FE can surface the artifact name.
        """
        self._post(self._log_url, {
            'message': f"file: {key} -> {filepath}",
            'levelname': 'INFO',
            'created': time.time(),
        })

    def on_progress(self, pct: float, eta_seconds: int | None = None) -> None:
        """POST to V2 /progress/ with the actual schema (progress_pct, eta_seconds).

        Pass ``eta_seconds=None`` when ETA is unknown (e.g. pct==0); the V2
        endpoint accepts null per ``api_v2.py:1110-1116``.

        W3 (TASK-1926): also emits a periodic HEARTBEAT log-flush every
        ``HEARTBEAT_INTERVAL_S`` seconds so a long run (e.g. 21h) that dies
        near completion still leaves a recent partial log in ``Run.log`` — today
        the log only flushes on explicit status/metric callbacks.
        """
        try:
            self._post(self._progress_url, {
                'progress_pct': float(pct),
                'eta_seconds': int(eta_seconds) if eta_seconds is not None else None,
            })
        except Exception:
            logger.exception('on_progress POST failed')

        # W3 (TASK-1926): periodic heartbeat
        now = time.time()
        if now - self._last_heartbeat_t >= self.HEARTBEAT_INTERVAL_S:
            try:
                self._post(self._log_url, {
                    'message': (
                        f"heartbeat: {pct:.1f}% complete"
                        + (f", eta={eta_seconds}s" if eta_seconds is not None else '')
                    ),
                    'levelname': 'DEBUG',
                    'created': now,
                })
                self._last_heartbeat_t = now
            except Exception:
                logger.debug('on_progress heartbeat POST failed', exc_info=True)

    def on_mesh_features_ready(self) -> None:
        """No-op on HydrataCallback — the early partial emit is wired by run_and_report.

        The sampler is not accessible here; ``run_and_report`` wraps the callback
        with an ``_EarlyPartialCallback`` that has the sampler reference (TASK-1924).
        """

    @classmethod
    def from_config(
        cls,
        scenario_config: dict,
    ) -> "HydrataCallback":
        """Convenience constructor from a scenario config dict.

        TASK-1049: signature dropped ``username``/``password`` — the token
        is read from the environment, not threaded through arguments.
        Raises ``KeyError`` if any required field is missing — fail-fast
        beats silently POSTing to ``/api/v2/anuga/runs/0/`` for 404s.
        """
        required = ('control_server', 'project', 'id', 'run_id')
        missing = [k for k in required if not scenario_config.get(k)]
        if missing:
            raise KeyError(
                f'scenario_config missing required field(s): {missing}'
            )
        return cls(
            control_server=scenario_config['control_server'],
            project=scenario_config['project'],
            scenario=scenario_config['id'],
            run_id=scenario_config['run_id'],
        )
