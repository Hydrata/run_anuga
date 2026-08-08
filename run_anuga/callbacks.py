"""
Simulation callback protocol and implementations.

Callbacks provide structured progress reporting for ``run_sim()``:

* **NullCallback** — does nothing (default for standalone use).
* **LoggingCallback** — logs progress via Python logging (CLI mode).
* **TelemetryCallback** — the ONE web-reporting adapter: maps this protocol
  onto typed events on the epic-2662 telemetry endpoint.

TASK-2681 (epic 2662 W4.1) deleted ``HydrataCallback``, the per-channel
POSTer that talked to ``/api/v2/anuga/runs/<id>/{log,progress}/``. Both
routes are 410 tombstones; ``TelemetryCallback`` is the only web reporter.
"""

from __future__ import annotations

import logging
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
        # Skip percentage updates — they are immediately followed by a more
        # detailed logger.info line in the evolve loop.
        if status.endswith("%"):
            return
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


class TelemetryCallback:
    """SimulationCallback -> events-dialect adapter (TASK-2672, epic 2662 D4).

    Maps the legacy per-channel callback protocol onto the ONE typed-events
    endpoint via a :class:`gn_anuga.batch_common.telemetry_client.TelemetryClient`.
    The server folds the events into the canonical Process row and fans out
    to the Run during the migration window (D6), so the legacy /log/ +
    /progress/ POSTs this replaces stay behaviourally covered.

    Owns NO transport and NO construction logic: the client is constructed
    ONCE, explicitly, in ``run_anuga._handoff.run_and_report`` (the W0.1
    lesson from TASK-2663 — no sentinel-gated construction site a wrapper
    can shadow) and handed in. The client is fail-open after construction
    (every post returns bool, never raises), so nothing here can break the
    run loop. Rank-0 only by construction (the client only exists on rank 0).
    """

    def __init__(self, client):
        self.client = client

    def on_status(self, status: str, **kwargs: Any) -> None:
        """State words ride the log-event channel (folded into the bounded
        Process.log + fanned out to Run.log), mirroring the legacy dialect's
        'status: <word>' Run.log lines. Terminal transitions stay owned by
        the orchestrator/server."""
        self.client.log(f"status: {status}")

    def on_metric(self, key: str, value: Any) -> None:
        self.client.metric({key: value})

    def on_file(self, key: str, filepath: str) -> None:
        self.client.log(f"file: {key} -> {filepath}")

    def on_progress(self, pct: float, eta_seconds: int | None = None) -> None:
        """Progress event. eta_seconds is normally None — the server derives
        ETA from progress history (D7); a non-None value is the documented
        container override and is passed through."""
        self.client.progress(pct, eta_seconds=eta_seconds)

    def on_mesh_features_ready(self) -> None:
        """No-op — the early partial ledger emit is wired by run_and_report's
        wrapper, which has the sampler reference (TASK-1924)."""

    def close(self) -> None:
        """No-op — the client (and its watchdog thread) lifecycle is owned by
        run_and_report, which arms it before run_sim and stops it after the
        handoff."""
