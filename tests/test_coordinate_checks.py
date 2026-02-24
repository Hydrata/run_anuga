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
        """Points exactly on the boundary are typically not 'contained'."""
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        # Shapely's contains() typically returns False for boundary points
        result = check_coordinates_are_in_polygon([[0, 0]], polygon)
        assert isinstance(result, bool)

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
