"""Shared logging helper for run_anuga (TASK-1276).

anuga_core (``anuga/utilities/log.py``) installs a *root* logger formatter that
references ``%(mname)s`` / ``%(lnum)s`` (its own non-standard LogRecord fields).
run_anuga's module loggers propagate their records up to that root handler, but
run_anuga records don't carry ``mname`` / ``lnum``, so the root formatter raises
``KeyError: 'mname'`` on every emit, flooding CloudWatch with
``--- Logging error ---`` tracebacks (observed in TASK-1182 W2 canary 19).

Disabling propagation would fix the spam but would also hide run_anuga records
from pytest's ``caplog`` (which captures via a handler on the *root* logger),
breaking the suite's log-assertion tests. Instead we stamp the two fields onto
each record with a filter so anuga_core's formatter can render them. The filter
never drops a record; it only supplies the standard ``module`` / ``lineno`` as
the ``mname`` / ``lnum`` defaults when absent.
"""

from __future__ import annotations

import logging


class MnameLnumFilter(logging.Filter):
    """Stamp anuga_core's ``mname`` / ``lnum`` record fields when missing."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, 'mname'):
            record.mname = record.module
        if not hasattr(record, 'lnum'):
            record.lnum = record.lineno
        return True


# One shared instance is safe to attach to several loggers: a logger's filters
# run in its own ``handle()`` (before the record propagates to ancestor
# handlers), and ``filter()`` is idempotent.
_mname_lnum_filter = MnameLnumFilter()


def install_mname_filter(logger: logging.Logger) -> None:
    """Attach the shared mname/lnum filter to ``logger`` (idempotent)."""
    if not any(isinstance(f, MnameLnumFilter) for f in logger.filters):
        logger.addFilter(_mname_lnum_filter)
