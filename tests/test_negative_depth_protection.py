"""TASK-2226 — defense-in-depth for run 1283's negative-inlet-volume assert.

run_anuga/run.py runs ANUGA's own protect_against_infinitesimal_and_negative_
heights() on rank 0 AFTER the Raised-structure elevation correction + stage=0.0
init and BEFORE distribute(), so the PARALLEL Parallel_Inlet_operator never
sees a negative inlet volume at the first evolve step (run 1283: MPI_ABORT rank
10, anuga/parallel/parallel_inlet_operator.py:121). The SERIAL path already got
this for free (evolve protects before the operator's first __call__); this
closes the gap for the parallel path.

These tests exercise the REAL helper (run_utils.apply_negative_depth_protection)
that run.py calls, driven through the exact run.py rank-0 ordering on a tiny
synthetic domain. The original root-cause reproduction lives at
deploy/docs/reports/2026-07-11-repro-2213-inlet-assert.py.

Fixture geometry:
    elevation = -0.3 (wet)  ->  Raised +5 m over the inlet  ->  stage = 0.0
      => the inlet region is dry-above-stage => get_total_water_volume() < 0

Requires ANUGA.
"""
import pytest

pytest.importorskip("anuga")

# The inlet line falls inside the Raised footprint (domain centre).
_INLET_LINE = [[45.0, 40.0], [45.0, 60.0]]
_RAISED_POLYGON = [[35, 30], [65, 30], [65, 70], [35, 70], [35, 30]]


def _build_domain(with_raised_structure):
    """Drive the EXACT run.py rank-0 ordering: elevation -> Raised correction
    -> friction -> stage=0.0."""
    import anuga
    from run_anuga.run_utils import (
        make_raised_elevation_pairs,
        apply_raised_elevation_correction,
    )

    points, vertices, boundary = anuga.rectangular_cross(20, 20, len1=100.0, len2=100.0)
    domain = anuga.Domain(points, vertices, boundary)
    domain.set_name('test_2226')
    domain.set_flow_algorithm('DE0')
    # Bed BELOW stage=0.0 everywhere (legitimately wet before any raise).
    domain.set_quantity('elevation', -0.3, location='centroids')

    if with_raised_structure:
        input_data = {
            'scenario_config': {},
            'structure': {'features': [{
                'properties': {'method': 'Raised'},
                'geometry': {'type': 'Polygon', 'coordinates': [_RAISED_POLYGON]},
            }]},
        }
        apply_raised_elevation_correction(domain, make_raised_elevation_pairs(input_data))

    domain.set_quantity('friction', 0.03)
    domain.set_quantity('stage', 0.0, verbose=False)          # run.py:~268
    domain.set_quantities_to_be_stored(None)
    Br = anuga.Reflective_boundary(domain)
    domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})
    return domain


def _inlet_volume(domain):
    from anuga.structures.inlet import Inlet
    return Inlet(domain, _INLET_LINE, verbose=False).get_total_water_volume()


@pytest.mark.requires_anuga
class TestNegativeDepthProtection:

    def test_raised_inlet_is_negative_before_protection(self):
        """Precondition: without protection the inlet volume is negative — the
        run 1283 trigger. Proves the test actually exercises the bug."""
        domain = _build_domain(with_raised_structure=True)
        assert _inlet_volume(domain) < 0.0

    def test_protection_clips_raised_inlet_to_non_negative(self):
        """The fix: apply_negative_depth_protection clips the inlet volume to
        >= 0, so Parallel_Inlet_operator's `assert current_volume >= 0` holds."""
        from run_anuga.run_utils import apply_negative_depth_protection
        domain = _build_domain(with_raised_structure=True)
        fired = apply_negative_depth_protection(domain)
        assert fired is True
        assert _inlet_volume(domain) >= 0.0

    def test_inlet_operator_evolves_without_assert_after_protection(self):
        """End-to-end: an Inlet_operator over the raised region + evolve no
        longer raises the negative-volume AssertionError (pre-fix: run 1283)."""
        import anuga
        from run_anuga.run_utils import apply_negative_depth_protection
        domain = _build_domain(with_raised_structure=True)
        apply_negative_depth_protection(domain)
        anuga.Inlet_operator(domain, _INLET_LINE, Q=1.0, verbose=False)
        for _ in domain.evolve(yieldstep=0.5, finaltime=1.0):
            pass  # must not assert

    def test_no_raised_structure_control_is_noop(self):
        """CONTROL: without a Raised structure the inlet is already wet
        (elevation -0.3 < stage 0.0), so the whole domain is at-or-above bed and
        protection does NOT fire (no behavior change) — rules out 'any inlet
        always asserts'."""
        from run_anuga.run_utils import apply_negative_depth_protection
        domain = _build_domain(with_raised_structure=False)
        assert _inlet_volume(domain) >= 0.0
        fired = apply_negative_depth_protection(domain)
        assert fired is False
        assert _inlet_volume(domain) >= 0.0
