"""Stub culvert operator (TASK-3638, epic 3585 W2b).

Logs each culvert so the product slice runs end to end; it moves NO water.
The coupling engine replaces it in W3 (TASK-3642). anuga is imported lazily
so this module stays importable without the sim stack.
"""

from __future__ import annotations

import logging

from run_anuga.culverts.constants import BLOCKAGE_METHOD

logger = logging.getLogger(__name__)


def make_stub_operator(domain, culverts, log=None):
    """Attach a no-op fractional-step operator that names every culvert once."""
    from anuga.operators.base_operator import Operator

    log = log or logger

    class CulvertStubOperator(Operator):
        def __init__(self, domain, culverts):
            super().__init__(domain, description="culvert stub (no exchange)", label="culvert_stub")
            self.culverts = list(culverts)
            self.calls = 0
            if getattr(domain, "processor", 0) == 0:
                for culvert in self.culverts:
                    log.info(
                        "culvert stub operator: %s, %s x%d, footprint %.2f m, length %.2f m, "
                        "blockage %.0f%% (%s); no water is exchanged until the culvert engine ships",
                        culvert.label, culvert.shape, culvert.barrels, culvert.footprint_width_m,
                        culvert.length_m, culvert.blockage_pct, BLOCKAGE_METHOD,
                    )

        def __call__(self):
            self.calls += 1

        def parallel_safe(self):
            return True

        def statistics(self):
            return f"Culvert stub operator: {len(self.culverts)} culvert(s), no exchange"

        def timestepping_statistics(self):
            return f"Culvert stub operator: {self.calls} call(s)"

    return CulvertStubOperator(domain, culverts)
