"""Tests for run_anuga.defaults — verify constant types and sensible ranges."""

from run_anuga import defaults


def test_building_burn_height():
    assert isinstance(defaults.BUILDING_BURN_HEIGHT_M, (int, float))
    assert 0 < defaults.BUILDING_BURN_HEIGHT_M < 100


def test_building_mannings_n():
    assert isinstance(defaults.BUILDING_MANNINGS_N, (int, float))
    assert defaults.BUILDING_MANNINGS_N > 0


def test_default_mannings_n():
    assert isinstance(defaults.DEFAULT_MANNINGS_N, (int, float))
    assert 0 < defaults.DEFAULT_MANNINGS_N < 1


def test_rainfall_factor():
    # 1 mm/hr = 1/1000 m/hr = 1/1000/3600 m/s = 2.7778e-7 m/s
    import math
    expected = 1.0 / (1000.0 * 3600.0)
    assert math.isclose(defaults.RAINFALL_FACTOR, expected, rel_tol=1e-9)


def test_minimum_storable_height():
    assert isinstance(defaults.MINIMUM_STORABLE_HEIGHT_M, float)
    assert defaults.MINIMUM_STORABLE_HEIGHT_M > 0


def test_min_allowed_height():
    assert isinstance(defaults.MIN_ALLOWED_HEIGHT_M, float)
    assert defaults.MIN_ALLOWED_HEIGHT_M > 0
    assert defaults.MIN_ALLOWED_HEIGHT_M < defaults.MINIMUM_STORABLE_HEIGHT_M


def test_yieldstep_limits():
    assert isinstance(defaults.MAX_YIELDSTEPS, int)
    assert defaults.MAX_YIELDSTEPS > 0
    assert isinstance(defaults.MIN_YIELDSTEP_S, int)
    assert isinstance(defaults.MAX_YIELDSTEP_S, int)
    assert defaults.MIN_YIELDSTEP_S < defaults.MAX_YIELDSTEP_S


def test_max_triangle_area():
    assert isinstance(defaults.MAX_TRIANGLE_AREA, int)
    assert defaults.MAX_TRIANGLE_AREA > 0


def test_k_nearest_neighbours():
    assert isinstance(defaults.K_NEAREST_NEIGHBOURS, int)
    assert defaults.K_NEAREST_NEIGHBOURS >= 1


def test_default_mesher_exe():
    assert isinstance(defaults.DEFAULT_MESHER_EXE, str)
    assert defaults.DEFAULT_MESHER_EXE.endswith("mesher")
