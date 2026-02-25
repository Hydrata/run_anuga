"""
Simulation logging configuration.

Provides three public functions that manage logging for run_anuga simulations:

* ``configure_simulation_logging()`` — adds a FileHandler (and optionally a
  StreamHandler) to the **root logger** so that both run_anuga's own messages
  *and* ANUGA's ``log.critical()`` output land in the same log file.

* ``neutralize_anuga_logging()`` — prevents ANUGA from calling
  ``logging.basicConfig()`` and creating stray ``anuga_*.log`` files.

* ``teardown_simulation_logging()`` — removes only the handlers we added,
  restoring the root logger to its previous state.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Sentinel attribute added to handlers we create, so teardown can find them.
_HANDLER_TAG = "_run_anuga_handler"

# Stashed original root level, set by configure and restored by teardown.
_original_root_level: int | None = None


def _is_django_configured() -> bool:
    """Return True if Django settings are loaded and have a LOGGING dict."""
    try:
        from django.conf import settings

        return bool(getattr(settings, "LOGGING", None))
    except Exception:
        return False


def configure_simulation_logging(
    output_dir: str,
    batch_number: int = 1,
    file_level: int = logging.INFO,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """Add a FileHandler (and optional StreamHandler) to the root logger.

    Parameters
    ----------
    output_dir:
        Directory for the log file.  Created if it doesn't exist.
    batch_number:
        Appended to the filename: ``run_anuga_{batch_number}.log``.
    file_level:
        Logging level for the file handler.
    console_level:
        Logging level for the console handler (ignored in Django mode).

    Returns
    -------
    logging.Logger
        A named logger ``run_anuga.sim`` for the caller to use.
    """
    global _original_root_level

    root = logging.getLogger()

    # Idempotent: remove any previously tagged handlers first.
    _remove_tagged_handlers(root)

    # Stash original root level so teardown can restore it.
    _original_root_level = root.level

    # Ensure output directory exists.
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- File handler ---
    log_path = os.path.join(output_dir, f"run_anuga_{batch_number}.log")
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    setattr(file_handler, _HANDLER_TAG, True)
    root.addHandler(file_handler)

    django_mode = _is_django_configured()

    if not django_mode:
        # --- Console handler (CLI only) ---
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(
            logging.Formatter("%(levelname)s %(message)s")
        )
        setattr(console_handler, _HANDLER_TAG, True)
        root.addHandler(console_handler)

        # Set root level to the more verbose of the two so both handlers see
        # the messages they've subscribed to.
        root.setLevel(min(file_level, console_level))
    # In Django mode: don't touch root level or add a console handler.

    return logging.getLogger("run_anuga.sim")


def neutralize_anuga_logging(output_dir: str) -> None:
    """Prevent ANUGA from running ``logging.basicConfig()`` and creating stray log files.

    Must be called **after** ``import anuga`` but **before** any
    ``anuga.utilities.log.*()`` call fires (i.e. before ``domain.evolve()``).

    Safe no-op when ANUGA is not installed.
    """
    try:
        import anuga.utilities.log as anuga_log
    except ImportError:
        return

    # Tell ANUGA its logging is already set up.
    anuga_log._setup = True

    # Redirect ANUGA's internal log file into the output directory instead of CWD.
    anuga_log.log_filename = os.path.join(output_dir, "anuga_internal.log")

    # Suppress ANUGA's own console handler by setting threshold above CRITICAL.
    anuga_log.console_logging_level = logging.CRITICAL + 1


def teardown_simulation_logging() -> None:
    """Remove tagged handlers from the root logger and restore its level.

    Idempotent — safe to call multiple times or when configure was never called.
    """
    global _original_root_level

    root = logging.getLogger()
    _remove_tagged_handlers(root)

    if _original_root_level is not None:
        root.setLevel(_original_root_level)
        _original_root_level = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _remove_tagged_handlers(logger: logging.Logger) -> None:
    """Remove all handlers that carry our ``_run_anuga_handler`` tag."""
    for handler in logger.handlers[:]:
        if getattr(handler, _HANDLER_TAG, False):
            logger.removeHandler(handler)
            handler.close()
