"""Tests for run_anuga.defaults — verify constant types and physically-justified ranges."""

import math

from run_anuga import defaults


def test_building_burn_height():
    assert isinstance(defaults.BUILDING_BURN_HEIGHT_M, (int, float))
    # Typical building height 2–15 m; 5 m is standard for flood modelling
    assert 2.0 <= defaults.BUILDING_BURN_HEIGHT_M <= 15.0


def test_building_mannings_n():
    assert isinstance(defaults.BUILDING_MANNINGS_N, (int, float))
    # Must be >> 1 to effectively block flow through building footprints
    assert defaults.BUILDING_MANNINGS_N > 1.0


def test_default_mannings_n():
    assert isinstance(defaults.DEFAULT_MANNINGS_N, (int, float))
    # Sealed urban/short grass: 0.035–0.050; 0.04 is the ANUGA default
    assert 0.035 <= defaults.DEFAULT_MANNINGS_N <= 0.050


def test_rainfall_factor():
    # 1 mm/hr = 1/1000 m/hr = 1/1000/3600 m/s = 2.7778e-7 m/s
    expected = 1.0 / (1000.0 * 3600.0)
    assert math.isclose(defaults.RAINFALL_FACTOR, expected, rel_tol=1e-9)
    # Sanity check: 100 mm/hr (heavy rainfall) → ~2.78e-5 m/s
    assert math.isclose(100 * defaults.RAINFALL_FACTOR, 100 / (1000.0 * 3600.0), rel_tol=1e-9)


def test_minimum_storable_height():
    assert isinstance(defaults.MINIMUM_STORABLE_HEIGHT_M, float)
    # 1–5 mm is the practical range; 5 mm (0.005) is the standard ANUGA default
    assert 0.001 <= defaults.MINIMUM_STORABLE_HEIGHT_M <= 0.005


def test_min_allowed_height():
    assert isinstance(defaults.MIN_ALLOWED_HEIGHT_M, float)
    # ANUGA's internal wet/dry threshold is 1e-5 m
    assert defaults.MIN_ALLOWED_HEIGHT_M == 1.0e-05
    assert defaults.MIN_ALLOWED_HEIGHT_M < defaults.MINIMUM_STORABLE_HEIGHT_M


def test_yieldstep_limits():
    assert isinstance(defaults.MAX_YIELDSTEPS, int)
    # 50–200 yield steps is practical for most simulations
    assert 50 <= defaults.MAX_YIELDSTEPS <= 200

    assert isinstance(defaults.MIN_YIELDSTEP_S, int)
    # At most every 30–120 seconds for short simulations
    assert 30 <= defaults.MIN_YIELDSTEP_S <= 120

    assert isinstance(defaults.MAX_YIELDSTEP_S, int)
    # At least every 5–60 minutes for long simulations
    assert 300 <= defaults.MAX_YIELDSTEP_S <= 3600

    assert defaults.MIN_YIELDSTEP_S < defaults.MAX_YIELDSTEP_S


def test_max_triangle_area():
    assert isinstance(defaults.MAX_TRIANGLE_AREA, int)
    assert defaults.MAX_TRIANGLE_AREA > 0


def test_k_nearest_neighbours():
    assert isinstance(defaults.K_NEAREST_NEIGHBOURS, int)
    # 3 is the minimum for spatial interpolation; 8 is practical upper bound
    assert 3 <= defaults.K_NEAREST_NEIGHBOURS <= 8


def test_default_mesher_exe():
    assert isinstance(defaults.DEFAULT_MESHER_EXE, str)
    assert defaults.DEFAULT_MESHER_EXE.endswith("mesher")
