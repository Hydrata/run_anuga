"""Tests for correction_for_polar_quadrants() from run_utils.py."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from run_anuga.run_utils import correction_for_polar_quadrants


class TestCorrectionForPolarQuadrants:
    def test_first_quadrant(self):
        # Positive x, positive y -> 0
        assert correction_for_polar_quadrants(1.0, 1.0) == 0

    def test_second_quadrant(self):
        # Negative x, positive y -> pi
        assert correction_for_polar_quadrants(-1.0, 1.0) == pytest.approx(math.pi)

    def test_third_quadrant(self):
        # Negative x, negative y -> pi
        assert correction_for_polar_quadrants(-1.0, -1.0) == pytest.approx(math.pi)

    def test_fourth_quadrant(self):
        # Positive x, negative y -> 2*pi
        assert correction_for_polar_quadrants(1.0, -1.0) == pytest.approx(2 * math.pi)

    def test_zero_base_positive_height(self):
        # base=0, height>0: neither base>0 nor base<0, so result stays 0
        assert correction_for_polar_quadrants(0.0, 1.0) == 0

    def test_zero_base_negative_height(self):
        # base=0, height<0: neither base>0 nor base<0, so result stays 0
        assert correction_for_polar_quadrants(0.0, -1.0) == 0

    def test_positive_base_zero_height(self):
        # base>0, height=0: neither height>0 nor height<0, so result stays 0
        assert correction_for_polar_quadrants(1.0, 0.0) == 0

    def test_negative_base_zero_height(self):
        # base<0, height=0: neither height>0 nor height<0, so result stays 0
        assert correction_for_polar_quadrants(-1.0, 0.0) == 0

    def test_large_values(self):
        assert correction_for_polar_quadrants(1e6, 1e6) == 0
        assert correction_for_polar_quadrants(-1e6, 1e6) == pytest.approx(math.pi)

    def test_small_values(self):
        assert correction_for_polar_quadrants(1e-10, 1e-10) == 0
        assert correction_for_polar_quadrants(-1e-10, -1e-10) == pytest.approx(math.pi)


class TestCorrectionForPolarQuadrantsProperty:
    @given(
        base=st.floats(min_value=0.01, max_value=1e6),
        height=st.floats(min_value=0.01, max_value=1e6),
    )
    def test_first_quadrant_always_zero(self, base, height):
        assert correction_for_polar_quadrants(base, height) == 0

    @given(
        base=st.floats(min_value=-1e6, max_value=-0.01),
        height=st.floats(min_value=0.01, max_value=1e6),
    )
    def test_second_quadrant_always_pi(self, base, height):
        assert correction_for_polar_quadrants(base, height) == pytest.approx(math.pi)

    @given(
        base=st.floats(min_value=-1e6, max_value=-0.01),
        height=st.floats(min_value=-1e6, max_value=-0.01),
    )
    def test_third_quadrant_always_pi(self, base, height):
        assert correction_for_polar_quadrants(base, height) == pytest.approx(math.pi)

    @given(
        base=st.floats(min_value=0.01, max_value=1e6),
        height=st.floats(min_value=-1e6, max_value=-0.01),
    )
    def test_fourth_quadrant_always_2pi(self, base, height):
        assert correction_for_polar_quadrants(base, height) == pytest.approx(2 * math.pi)

    @given(
        base=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        height=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_result_is_valid_angle_component(self, base, height):
        result = correction_for_polar_quadrants(base, height)
        assert result in (0, pytest.approx(math.pi), pytest.approx(2 * math.pi))
