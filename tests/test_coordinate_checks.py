"""Component tests for check_coordinates_are_in_polygon().

Requires shapely (marked requires_geo).
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from run_anuga.run_utils import check_coordinates_are_in_polygon


@pytest.mark.requires_geo
class TestCheckCoordinatesAreInPolygon:
    def test_point_inside(self):
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        assert check_coordinates_are_in_polygon([[5, 5]], polygon) is True

    def test_point_outside(self):
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        assert check_coordinates_are_in_polygon([[15, 15]], polygon) is False

    def test_single_point_not_nested(self):
        """Single point [x, y] instead of [[x, y]]."""
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        assert check_coordinates_are_in_polygon([5.0, 5.0], polygon) is True

    def test_multiple_points_all_inside(self):
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        coords = [[1, 1], [5, 5], [9, 9]]
        assert check_coordinates_are_in_polygon(coords, polygon) is True

    def test_multiple_points_one_outside(self):
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        coords = [[1, 1], [15, 15]]
        assert check_coordinates_are_in_polygon(coords, polygon) is False

    def test_point_on_edge(self):
        """Points exactly on the boundary return False per Shapely contains() semantics."""
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        # Shapely's contains() returns False for boundary points (interior test only)
        assert check_coordinates_are_in_polygon([[0, 0]], polygon) is False

    def test_point_near_center(self):
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        assert check_coordinates_are_in_polygon([[5, 5]], polygon) is True

    def test_triangle_polygon(self):
        polygon = [[0, 0], [10, 0], [5, 10]]
        assert check_coordinates_are_in_polygon([[4, 3]], polygon) is True
        assert check_coordinates_are_in_polygon([[0, 10]], polygon) is False

    def test_large_coordinates(self):
        """Real-world UTM coordinates."""
        polygon = [[321000, 5812000], [321100, 5812000], [321100, 5812100], [321000, 5812100]]
        assert check_coordinates_are_in_polygon([[321050, 5812050]], polygon) is True
        assert check_coordinates_are_in_polygon([[320000, 5812050]], polygon) is False

    def test_closed_ring_raises_clear_error_not_shapely_crash(self):
        """TASK-2187 (epic 2147 W2) — a CLOSED RING (e.g. a Rainfall Polygon's
        outer ring mistakenly routed through the Surface-inflow geometry
        path, ~run_utils.py:1340) must raise a CLEAR, named error instead of
        the opaque shapely 'Point() takes only scalar or 1-size vector
        arguments' crash. Silently returning False is NOT acceptable either
        (that would silently skip registering the operator with no signal —
        the same 'never warn-and-continue' guard as the 2155 NaN guard)."""
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        closed_ring = [
            [251000.0, 6271800.0], [251200.0, 6271800.0], [251200.0, 6272000.0],
            [251000.0, 6272000.0], [251000.0, 6271800.0],
        ]
        nested_ring = [closed_ring]  # e.g. a raw GeoJSON Polygon.coordinates fallthrough
        with pytest.raises(ValueError) as excinfo:
            check_coordinates_are_in_polygon(nested_ring, polygon)
        # A controlled, named error — NOT the raw shapely ValueError.
        assert 'Point()' not in str(excinfo.value)

    def test_multilinestring_flattened_points_still_work(self):
        """No regression in the TASK-1113 MultiLineString inflow path: a
        flattened list of flat [x, y] points (what _flatten_line_coordinates
        returns for MultiLineString/LineString) still checks fine."""
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        flattened_multilinestring_points = [[1, 1], [2, 2], [3, 3], [9, 9]]
        assert check_coordinates_are_in_polygon(flattened_multilinestring_points, polygon) is True


@pytest.mark.requires_geo
class TestCheckCoordinatesProperty:
    @given(
        x=st.floats(min_value=0.1, max_value=9.9),
        y=st.floats(min_value=0.1, max_value=9.9),
    )
    def test_interior_points_always_inside(self, x, y):
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        assert check_coordinates_are_in_polygon([[x, y]], polygon) is True

    @given(
        x=st.floats(min_value=11, max_value=100),
        y=st.floats(min_value=11, max_value=100),
    )
    def test_exterior_points_always_outside(self, x, y):
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        assert check_coordinates_are_in_polygon([[x, y]], polygon) is False
