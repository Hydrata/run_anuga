"""Tests for ``generate_stac`` early-return guard.

Pins the deterministic no-op behavior introduced when
``initial_time_iso_string`` or ``output_quantities`` is falsy. Before this
guard, ``datetime.fromisoformat(None)`` and the for-loop over
``output_quantities`` would raise on every legacy run (older scenario
packages do not carry ``initial_time`` / ``output_quantities`` in
``scenario_config``). The guard converts the silent-fail into a logged
no-op so operators can spot the coverage gap.
"""

import logging

import pytest

from run_anuga.run_utils import generate_stac


class TestGenerateStacEarlyReturn:
    """If the required scenario_config inputs are missing, ``generate_stac``
    returns without touching boto3/pystac and logs one info line."""

    def test_returns_none_when_initial_time_missing(self, caplog):
        """initial_time_iso_string=None must short-circuit before AWS lookup."""
        with caplog.at_level(logging.INFO, logger='run_anuga.run_utils'):
            result = generate_stac(
                output_directory='/tmp/nowhere',
                run_label='run_legacy',
                output_quantities=['stage', 'depth'],
                initial_time_iso_string=None,
            )
        assert result is None
        assert any('STAC generation skipped' in r.message for r in caplog.records), (
            f"expected skip log, got: {[r.message for r in caplog.records]}"
        )

    def test_returns_none_when_output_quantities_missing(self, caplog):
        """output_quantities=None must short-circuit before AWS lookup."""
        with caplog.at_level(logging.INFO, logger='run_anuga.run_utils'):
            result = generate_stac(
                output_directory='/tmp/nowhere',
                run_label='run_legacy',
                output_quantities=None,
                initial_time_iso_string='2026-05-15T00:00:00Z',
            )
        assert result is None
        assert any('STAC generation skipped' in r.message for r in caplog.records)

    def test_returns_none_when_output_quantities_empty_list(self, caplog):
        """Empty list (falsy) is treated the same as None."""
        with caplog.at_level(logging.INFO, logger='run_anuga.run_utils'):
            result = generate_stac(
                output_directory='/tmp/nowhere',
                run_label='run_legacy',
                output_quantities=[],
                initial_time_iso_string='2026-05-15T00:00:00Z',
            )
        assert result is None
        assert any('STAC generation skipped' in r.message for r in caplog.records)

    def test_returns_none_when_initial_time_empty_string(self, caplog):
        """Empty string (falsy) is treated the same as None."""
        with caplog.at_level(logging.INFO, logger='run_anuga.run_utils'):
            result = generate_stac(
                output_directory='/tmp/nowhere',
                run_label='run_legacy',
                output_quantities=['stage'],
                initial_time_iso_string='',
            )
        assert result is None
        assert any('STAC generation skipped' in r.message for r in caplog.records)

    def test_log_includes_run_label_for_diagnostics(self, caplog):
        """The log line must include run_label so operators can trace which
        run triggered the no-op."""
        with caplog.at_level(logging.INFO, logger='run_anuga.run_utils'):
            generate_stac(
                output_directory='/tmp/nowhere',
                run_label='run_42_diagnostic',
                output_quantities=None,
                initial_time_iso_string=None,
            )
        skip_lines = [r.message for r in caplog.records if 'STAC generation skipped' in r.message]
        assert skip_lines, "no skip-log emitted"
        assert 'run_42_diagnostic' in skip_lines[0], (
            f"run_label missing from log: {skip_lines[0]}"
        )
