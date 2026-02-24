"""Tests for compute_yieldstep() — yield step calculation logic."""

import math

from hypothesis import given
from hypothesis import strategies as st

from run_anuga import defaults
from run_anuga.run_utils import compute_yieldstep


class TestComputeYieldstep:
    def test_short_duration_clamped_to_min(self):
        """Duration < MAX_YIELDSTEPS * MIN_YIELDSTEP_S -> clamped to MIN."""
        result = compute_yieldstep(60)
        assert result == defaults.MIN_YIELDSTEP_S

    def test_long_duration_clamped_to_max(self):
        """Duration > MAX_YIELDSTEPS * MAX_YIELDSTEP_S -> clamped to MAX."""
        result = compute_yieldstep(1_000_000)
        assert result == defaults.MAX_YIELDSTEP_S

    def test_medium_duration_within_bounds(self):
        """Duration that produces yieldstep between min and max."""
        # 30000 / 100 = 300, which is between 60 and 1800
        result = compute_yieldstep(30000)
        assert result == 300
        assert defaults.MIN_YIELDSTEP_S <= result <= defaults.MAX_YIELDSTEP_S

    def test_exact_min_boundary(self):
        """Duration = MAX_YIELDSTEPS * MIN_YIELDSTEP_S."""
        duration = defaults.MAX_YIELDSTEPS * defaults.MIN_YIELDSTEP_S  # 6000
        result = compute_yieldstep(duration)
        assert result == defaults.MIN_YIELDSTEP_S

    def test_exact_max_boundary(self):
        """Duration = MAX_YIELDSTEPS * MAX_YIELDSTEP_S."""
        duration = defaults.MAX_YIELDSTEPS * defaults.MAX_YIELDSTEP_S  # 180000
        result = compute_yieldstep(duration)
        assert result == defaults.MAX_YIELDSTEP_S

    def test_just_above_min(self):
        """Duration just enough to produce base_step > MIN."""
        # Need floor(duration / 100) > 60, so duration > 6100
        result = compute_yieldstep(6100)
        assert result == 61

    def test_one_second_duration(self):
        """Very short duration clamped to MIN."""
        result = compute_yieldstep(1)
        assert result == defaults.MIN_YIELDSTEP_S

    def test_common_scenario_600s(self):
        """600s simulation (small_test default)."""
        result = compute_yieldstep(600)
        # 600 / 100 = 6, clamped up to MIN_YIELDSTEP_S (60)
        assert result == defaults.MIN_YIELDSTEP_S

    def test_common_scenario_1800s(self):
        """1800s simulation (examples/small_test)."""
        result = compute_yieldstep(1800)
        # 1800 / 100 = 18, clamped up to 60
        assert result == defaults.MIN_YIELDSTEP_S

    def test_common_scenario_86400s(self):
        """24-hour simulation."""
        result = compute_yieldstep(86400)
        # 86400 / 100 = 864
        assert result == 864

    def test_returns_integer(self):
        result = compute_yieldstep(30000)
        assert isinstance(result, int)


class TestComputeYieldstepProperty:
    @given(duration=st.integers(min_value=1, max_value=10_000_000))
    def test_always_within_bounds(self, duration):
        result = compute_yieldstep(duration)
        assert defaults.MIN_YIELDSTEP_S <= result <= defaults.MAX_YIELDSTEP_S

    @given(duration=st.integers(min_value=1, max_value=10_000_000))
    def test_monotonic_nondecreasing(self, duration):
        """Longer durations never produce shorter yieldsteps."""
        result = compute_yieldstep(duration)
        result_longer = compute_yieldstep(duration + 100)
        assert result_longer >= result

    @given(duration=st.integers(min_value=1, max_value=10_000_000))
    def test_matches_manual_calculation(self, duration):
        """Verify against the original inline logic."""
        base = math.floor(duration / defaults.MAX_YIELDSTEPS)
        expected = max(base, defaults.MIN_YIELDSTEP_S)
        expected = min(expected, defaults.MAX_YIELDSTEP_S)
        assert compute_yieldstep(duration) == expected
