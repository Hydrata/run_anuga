"""TASK-2156 (epic 2147 W2) — mass-balance e2e: the trust-chain proof.

One test kills two failure classes forever:
  * a wrong ``RAINFALL_FACTOR`` (~3.6x off — see the RED-PROOF note below), and
  * a broken model_start/timestamp alignment (2085/2147 W0 concern — a total
    mismatch either raises via the TASK-2155 NaN guard, or silently applies
    zero rain, both of which fail this test's volume assertion loudly).

Builds a SMALL, FLAT, CLOSED (all-Reflective-boundary) synthetic domain via
``anuga.rectangular_cross_domain`` — no DEM/mesh-region/GeoServer package
machinery, so the run is fast and the expected volume is exactly, trivially
computable. A closed domain with no negative (absorbing) rates loses no
mass to outflow, so ``domain.get_water_volume()`` at the end of the run
must equal the sum of every operator's injected volume, to numerical
roundoff.

TWO variants (see AC in the class docstrings below):
  (a) test_mass_balance_aligned_constant_inflow_plus_rain — exercises the
      REAL Inlet_operator/inflow-geometry code path (the function TASK-2187
      hardened) alongside the rain path.
  (b) test_mass_balance_rain_only — the permanent key-mode (rainfall-only,
      no inflow attached) fixture, proving TASK-2154's "inflow optional"
      contract has a real, physically-verified consumer downstream.

Both call ``apply_inflows_to_domain`` DIRECTLY with the REAL anuga operator
classes (not mocks) and evolve a REAL domain — this is the actual physics,
not a unit-test double.

RED-PROOF (TASK-2156 AC): temporarily setting
``run_anuga.defaults.RAINFALL_FACTOR = 1e-6`` (real value is
``1/(1000*3600)`` ~= 2.78e-7, so 1e-6 is ~3.6x larger — exactly the
"RAINFALL_FACTOR off ~3.6x" failure class this test exists to catch) makes
BOTH tests below FAIL (measured volume comes out ~3.6x too high vs the
still-correctly-computed expected volume); reverting restores GREEN. See
docs/2026-07-08-w2.3-mass-balance-redproof.md for the manual verification
transcript (this is not itself an automated test — the production
RAINFALL_FACTOR is a module constant, not a request-scoped knob).
"""
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = [pytest.mark.requires_anuga, pytest.mark.slow]


DOMAIN_SIDE_M = 10.0        # 10m x 10m square domain
DOMAIN_CELLS = 8            # 8x8 rectangular-cross cells -> 256 triangles, fast
DOMAIN_AREA_M2 = DOMAIN_SIDE_M * DOMAIN_SIDE_M
DURATION_S = 300            # 5 minutes model time
RAIN_MM_PER_HR = 40.0       # uniform hyetograph intensity
INFLOW_Q_M3_S = 0.001       # constant Surface inflow discharge
VOLUME_TOLERANCE = 0.02     # within 2%, per AC

# INDEPENDENT oracle — deliberately hardcoded, NOT imported from
# run_anuga.defaults.RAINFALL_FACTOR. If the expected-volume math derived
# its conversion factor from the SAME module constant the code under test
# applies, a regression that corrupts that constant would move both sides
# of the assertion together and the test would stay green while silently
# blind (this is exactly the ~3.6x RAINFALL_FACTOR failure class the AC
# calls out — verified via the RED-PROOF in the module docstring).
_EXPECTED_RAINFALL_FACTOR = 1.0 / (1000.0 * 3600.0)  # mm/hr -> m/s, m/mm, hr/s


def _build_closed_flat_domain():
    """A small, flat, fully-Reflective (closed — no mass leaves) domain."""
    anuga = pytest.importorskip("anuga")
    domain = anuga.rectangular_cross_domain(
        DOMAIN_CELLS, DOMAIN_CELLS, len1=DOMAIN_SIDE_M, len2=DOMAIN_SIDE_M,
    )
    domain.set_quantity('elevation', 0.0)
    domain.set_quantity('stage', 0.0)
    domain.set_quantity('friction', 0.01)
    # No SWW/checkpoint output needed for a pure volume-conservation check —
    # avoids littering the repo root with a stray domain.sww per test run.
    domain.set_store(False)
    reflective = anuga.Reflective_boundary(domain)
    domain.set_boundary({
        'left': reflective, 'right': reflective,
        'top': reflective, 'bottom': reflective,
    })
    return domain


def _rainfall_feature(start, duration_s, intensity_mm_per_hr):
    """A genuine timeseries (list-of-dicts) hyetograph, NOT a data_constant
    float — exercises run_utils._merge_timeseries' per-second model_start
    alignment path, not the constant-value shortcut. Every row carries the
    SAME intensity so the expected integral is exact and discretization-
    noise-free regardless of how ANUGA's adaptive internal timestep samples
    it (see module docstring)."""
    rows = [
        {'timestamp': start.isoformat(), 'value': intensity_mm_per_hr},
        {'timestamp': (start + timedelta(seconds=duration_s / 2)).isoformat(),
         'value': intensity_mm_per_hr},
        {'timestamp': (start + timedelta(seconds=duration_s)).isoformat(),
         'value': intensity_mm_per_hr},
    ]
    return {
        'type': 'Feature',
        'id': 'rain_uniform',
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[
                [0.0, 0.0], [DOMAIN_SIDE_M, 0.0],
                [DOMAIN_SIDE_M, DOMAIN_SIDE_M], [0.0, DOMAIN_SIDE_M], [0.0, 0.0],
            ]],
        },
        'properties': {'data': rows},
    }


def _inflow_feature(q_m3_s):
    return {
        'type': 'Feature',
        'id': 'inflow_const',
        'geometry': {
            'type': 'LineString',
            'coordinates': [[4.5, 4.5], [5.5, 5.5]],
        },
        # A scalar (non-list) `data` is the CONSTANT-inflow case (no
        # timeseries merge) — "constant-inflow" per the AC naming.
        'properties': {'data': q_m3_s},
    }


def _run_and_get_water_volume(rainfall_features, inflow_features, start, duration_s):
    import anuga
    from anuga.operators.rate_operators import Polygonal_rate_operator
    from run_anuga.run_utils import apply_inflows_to_domain

    domain = _build_closed_flat_domain()
    input_data = {
        'rainfall': {'features': rainfall_features},
        'inflow': {'features': inflow_features},
        'catchment': {'features': []},
        'boundary_polygon': [
            [0.0, 0.0], [DOMAIN_SIDE_M, 0.0],
            [DOMAIN_SIDE_M, DOMAIN_SIDE_M], [0.0, DOMAIN_SIDE_M],
        ],
    }
    apply_inflows_to_domain(
        input_data=input_data, domain=domain, start=start, duration=duration_s,
        Polygonal_rate_operator=Polygonal_rate_operator,
        Inlet_operator=anuga.Inlet_operator,
    )
    for _ in domain.evolve(yieldstep=duration_s / 3, finaltime=duration_s):
        pass
    return domain.get_water_volume()


def test_mass_balance_rain_only():
    """(b) rain-only, no inflow attached — the permanent key-mode fixture.
    Proves TASK-2154's 'inflow optional' contract downstream: a scenario
    with ONLY a rainfall hyetograph attached simulates and conserves mass
    exactly (within 2%)."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    measured_volume = _run_and_get_water_volume(
        rainfall_features=[_rainfall_feature(start, DURATION_S, RAIN_MM_PER_HR)],
        inflow_features=[],
        start=start, duration_s=DURATION_S,
    )
    expected_volume = RAIN_MM_PER_HR * _EXPECTED_RAINFALL_FACTOR * DOMAIN_AREA_M2 * DURATION_S
    assert measured_volume == pytest.approx(expected_volume, rel=VOLUME_TOLERANCE)


def test_mass_balance_aligned_constant_inflow_plus_rain():
    """(a) constant-inflow + rain (aligned) — exercises the REAL Surface
    inflow (Inlet_operator) geometry-check code path (check_coordinates_are_
    in_polygon, hardened by TASK-2187) alongside the rain path, in the SAME
    run. Total conserved volume = rain contribution + inflow contribution."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    measured_volume = _run_and_get_water_volume(
        rainfall_features=[_rainfall_feature(start, DURATION_S, RAIN_MM_PER_HR)],
        inflow_features=[_inflow_feature(INFLOW_Q_M3_S)],
        start=start, duration_s=DURATION_S,
    )
    expected_rain_volume = RAIN_MM_PER_HR * _EXPECTED_RAINFALL_FACTOR * DOMAIN_AREA_M2 * DURATION_S
    expected_inflow_volume = INFLOW_Q_M3_S * DURATION_S
    expected_volume = expected_rain_volume + expected_inflow_volume
    assert measured_volume == pytest.approx(expected_volume, rel=VOLUME_TOLERANCE)
