"""
Simulation callback protocol and implementations.

Callbacks provide structured progress reporting for ``run_sim()``.
They replace the scattered ``update_web_interface()`` calls with a
clean, swappable interface:

* **NullCallback** — does nothing (default for standalone use).
* **LoggingCallback** — logs progress via Python logging (CLI mode).
* **HydrataCallback** — HTTP PATCH to the Hydrata control server.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


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


class NullCallback:
    """Callback that silently discards all events.  Default for standalone use."""

    def on_status(self, status: str, **kwargs: Any) -> None:
        pass

    def on_metric(self, key: str, value: Any) -> None:
        pass

    def on_file(self, key: str, filepath: str) -> None:
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


class HydrataCallback:
    """
    Callback that reports progress to the Hydrata control server via HTTP PATCH.

    Parameters
    ----------
    username : str
        HTTP Basic Auth username.
    password : str
        HTTP Basic Auth password.
    control_server : str
        Base URL, e.g. ``"https://hydrata.com"``.
    project : int
        Project ID.
    scenario : int
        Scenario ID.
    run_id : int
        Run ID.
    """

    def __init__(
        self,
        username: str,
        password: str,
        control_server: str,
        project: int,
        scenario: int,
        run_id: int,
    ):
        self.username = username
        self.password = password
        self.control_server = control_server
        self.project = project
        self.scenario = scenario
        self.run_id = run_id

    @property
    def _url(self) -> str:
        base = self.control_server.rstrip("/") + "/"
        return f"{base}anuga/api/{self.project}/{self.scenario}/run/{self.run_id}/"

    def _patch(self, data: dict, files: dict | None = None) -> None:
        from run_anuga._imports import import_optional

        requests = import_optional("requests")
        client = requests.Session()
        client.auth = requests.auth.HTTPBasicAuth(self.username, self.password)
        data["project"] = self.project
        data["scenario"] = self.scenario
        response = client.patch(self._url, data=data, files=files)
        if response.status_code >= 400:
            logger.error(
                "Error updating web interface. HTTP code: %d - %s",
                response.status_code,
                response.text,
            )

    def on_status(self, status: str, **kwargs: Any) -> None:
        self._patch({"status": status})

    def on_metric(self, key: str, value: Any) -> None:
        self._patch({key: value})

    def on_file(self, key: str, filepath: str) -> None:
        with open(filepath, "rb") as f:
            self._patch({}, files={key: f})

    @classmethod
    def from_config(
        cls,
        username: str,
        password: str,
        scenario_config: dict,
    ) -> "HydrataCallback":
        """Convenience constructor from a scenario config dict."""
        return cls(
            username=username,
            password=password,
            control_server=scenario_config.get("control_server", ""),
            project=scenario_config.get("project", 0),
            scenario=scenario_config.get("id", 0),
            run_id=scenario_config.get("run_id", 0),
        )
