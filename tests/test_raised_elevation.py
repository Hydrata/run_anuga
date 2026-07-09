"""Regression tests for Raised-structure elevation seating.

TASK-2149 (F1): apply_raised_elevation_correction must use ABSOLUTE centroid
coordinates so the point-in-polygon test matches the absolute-UTM Raised polygons
regardless of the mesh geo_reference offset. The previous inline code used LOCAL
centroids (absolute=False), which silently matched ZERO centroids on any mesh with
a nonzero geo_reference offset (every local-offset mesh = every prod sim), so Raised
building heights were never applied.

Requires ANUGA.
"""
import numpy as np
import pytest

pytest.importorskip("anuga")


def _domain_with_offset(xll, yll):
    """A 100x100 m local-coordinate mesh with an explicit geo_reference offset.

    (xll, yll) != (0, 0) reproduces the local-offset regime in which the old
    absolute=False code silently dropped Raised structures.
    """
    import anuga
    points, vertices, boundary = anuga.rectangular_cross(20, 20, len1=100.0, len2=100.0)
    gr = anuga.Geo_reference(zone=56, xllcorner=xll, yllcorner=yll)
    domain = anuga.Domain(points, vertices, boundary, geo_reference=gr)
    domain.set_quantity("elevation", 10.0)
    return domain


# An absolute-UTM Raised footprint covering the middle of the domain.
def _abs_poly(xll, yll):
    return [
        [xll + 20, yll + 20], [xll + 80, yll + 20],
        [xll + 80, yll + 80], [xll + 20, yll + 80],
    ]


@pytest.mark.requires_anuga
class TestApplyRaisedElevationCorrection:
    def test_applies_on_nonzero_geo_reference_offset(self):
        """The regression case: a local-offset mesh. absolute=False would give 0 hits."""
        from run_anuga.run_utils import apply_raised_elevation_correction
        xll, yll = 382000.0, 6354000.0
        domain = _domain_with_offset(xll, yll)
        applied = apply_raised_elevation_correction(domain, [(_abs_poly(xll, yll), 5.0)])
        elev = domain.get_quantity("elevation").get_values(location="centroids")
        assert applied == 1, "Raised structure matched no centroids — absolute=False regression"
        assert elev.max() == pytest.approx(15.0), "footprint centroids not raised by 5 m"
        assert elev.min() == pytest.approx(10.0), "outside-footprint centroids must be untouched"
        assert (elev > 14.9).sum() > 0 and (elev < 10.1).sum() > 0

    def test_applies_on_zero_offset_mesh(self):
        """The no-hole (absolute-UTM) regime also works — offset-independent."""
        from run_anuga.run_utils import apply_raised_elevation_correction
        domain = _domain_with_offset(0.0, 0.0)
        applied = apply_raised_elevation_correction(domain, [(_abs_poly(0.0, 0.0), 3.0)])
        elev = domain.get_quantity("elevation").get_values(location="centroids")
        assert applied == 1
        assert elev.max() == pytest.approx(13.0)
        assert elev.min() == pytest.approx(10.0)

    def test_per_structure_heights(self):
        """Distinct heights applied per structure."""
        from run_anuga.run_utils import apply_raised_elevation_correction
        xll, yll = 382000.0, 6354000.0
        domain = _domain_with_offset(xll, yll)
        left = [[xll + 10, yll + 10], [xll + 40, yll + 10], [xll + 40, yll + 90], [xll + 10, yll + 90]]
        right = [[xll + 60, yll + 10], [xll + 90, yll + 10], [xll + 90, yll + 90], [xll + 60, yll + 90]]
        applied = apply_raised_elevation_correction(domain, [(left, 2.0), (right, 7.0)])
        elev = domain.get_quantity("elevation").get_values(location="centroids")
        assert applied == 2
        assert elev.max() == pytest.approx(17.0)   # right footprint: 10 + 7
        assert np.isclose(elev, 12.0).sum() > 0    # left footprint: 10 + 2

    def test_empty_and_degenerate_pairs_are_noops(self):
        from run_anuga.run_utils import apply_raised_elevation_correction
        domain = _domain_with_offset(382000.0, 6354000.0)
        assert apply_raised_elevation_correction(domain, []) == 0
        assert apply_raised_elevation_correction(domain, [([], 5.0)]) == 0
        elev = domain.get_quantity("elevation").get_values(location="centroids")
        assert elev.max() == pytest.approx(10.0), "no structure should have changed elevation"
