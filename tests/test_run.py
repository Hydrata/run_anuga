"""Tests for run_anuga.run — SIGALRM watchdog around MPI_Finalize.

The watchdog defends against the libmpi ompi_mpi_finalize -> usleep busy-loop
wedge observed in run 27593 forensics. Without it, a hanging MPI_Finalize call
would block the celery worker forever.
"""

import logging
import time
from unittest.mock import MagicMock

import pytest

from run_anuga.run import _finalize_with_timeout


class TestFinalizeWithTimeout:
    def test_fast_finalize_completes_without_warning(self, caplog):
        """A finalize() that returns immediately should not log a warning."""
        finalize = MagicMock(return_value=None)
        with caplog.at_level(logging.WARNING, logger="run_anuga.run"):
            _finalize_with_timeout(finalize, timeout_seconds=5)
        finalize.assert_called_once()
        # No warning should have been emitted on the happy path.
        assert "hung" not in caplog.text
        assert "MPI_Finalize" not in caplog.text

    def test_slow_finalize_times_out_and_logs_warning(self, caplog):
        """A finalize() that hangs should be abandoned and a warning logged."""
        def slow_finalize():
            time.sleep(60)

        with caplog.at_level(logging.WARNING, logger="run_anuga.run"):
            start = time.time()
            _finalize_with_timeout(slow_finalize, timeout_seconds=1)
            elapsed = time.time() - start

        # Watchdog must have returned within ~timeout, not after the full 60s sleep.
        assert elapsed < 5, f"watchdog took {elapsed}s, expected ~1s"
        # Warning must mention the hang and the OS-reclaim safety net.
        assert "MPI_Finalize hung" in caplog.text
        assert "OS will reclaim" in caplog.text

    def test_timeout_resets_alarm_and_handler(self):
        """After timeout, SIGALRM must be cancelled and original handler restored."""
        import signal

        sentinel_called = []

        def sentinel_handler(signum, frame):
            sentinel_called.append(True)

        previous = signal.signal(signal.SIGALRM, sentinel_handler)
        try:
            _finalize_with_timeout(lambda: time.sleep(60), timeout_seconds=1)
            # After return, the previous handler should be restored.
            current = signal.getsignal(signal.SIGALRM)
            assert current is sentinel_handler, (
                "previous SIGALRM handler was not restored"
            )
        finally:
            signal.signal(signal.SIGALRM, previous)

    def test_env_var_overrides_default_timeout(self, monkeypatch, caplog):
        """RUN_ANUGA_FINALIZE_TIMEOUT_SECONDS env var must be honoured."""
        monkeypatch.setenv("RUN_ANUGA_FINALIZE_TIMEOUT_SECONDS", "1")
        with caplog.at_level(logging.WARNING, logger="run_anuga.run"):
            start = time.time()
            _finalize_with_timeout(lambda: time.sleep(60))
            elapsed = time.time() - start
        assert elapsed < 5
        assert "1s" in caplog.text
