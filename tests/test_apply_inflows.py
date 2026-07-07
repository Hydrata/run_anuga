"""Tests for the W2.3 inflow/rainfall split in ``apply_inflows_to_domain``.

Polygonal_rate_operator (rainfall) is registered for every feature under
``input_data['rainfall']['features']``. Inlet_operator (surface inflow) is
registered for every feature under ``input_data['inflow']['features']``.
No more ``properties.type`` branching. Geometry IS the discriminator.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from run_anuga.run_utils import apply_inflows_to_domain


BOUNDARY_POLYGON = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def _rainfall_feature(fid='rain.1', data=1.0):
    return {
        'type': 'Feature',
        'id': fid,
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[
                [10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0], [10.0, 10.0],
            ]],
        },
        'properties': {'data': data},
    }


def _surface_feature(fid='surf.1', data=0.5):
    return {
        'type': 'Feature',
        'id': fid,
        'geometry': {
            'type': 'LineString',
            'coordinates': [[15.0, 15.0], [25.0, 15.0]],
        },
        'properties': {'data': data},
    }


def _input_data(rainfall_features=None, inflow_features=None):
    return {
        'rainfall': {'features': list(rainfall_features or [])},
        'inflow': {'features': list(inflow_features or [])},
        'catchment': {'features': []},
        'boundary_polygon': BOUNDARY_POLYGON,
    }


@pytest.fixture
def mocks():
    return {
        'domain': MagicMock(name='domain'),
        'Polygonal_rate_operator': MagicMock(name='Polygonal_rate_operator'),
        'Inlet_operator': MagicMock(name='Inlet_operator'),
    }


@pytest.fixture
def start():
    return datetime(2020, 1, 1, tzinfo=timezone.utc)


def test_rainfall_only_routes_to_polygonal_rate_operator(mocks, start):
    input_data = _input_data(rainfall_features=[_rainfall_feature()])
    apply_inflows_to_domain(
        input_data=input_data, domain=mocks['domain'], start=start, duration=60,
        Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
        Inlet_operator=mocks['Inlet_operator'],
    )
    assert mocks['Polygonal_rate_operator'].call_count == 1
    mocks['Inlet_operator'].assert_not_called()


def test_inflow_only_routes_to_inlet_operator(mocks, start):
    input_data = _input_data(inflow_features=[_surface_feature()])
    apply_inflows_to_domain(
        input_data=input_data, domain=mocks['domain'], start=start, duration=60,
        Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
        Inlet_operator=mocks['Inlet_operator'],
    )
    mocks['Polygonal_rate_operator'].assert_not_called()
    assert mocks['Inlet_operator'].call_count == 1


def test_rainfall_and_inflow_both_route_independently(mocks, start):
    input_data = _input_data(
        rainfall_features=[_rainfall_feature(fid='r1'), _rainfall_feature(fid='r2', data=2.0)],
        inflow_features=[_surface_feature(fid='s1'), _surface_feature(fid='s2', data=1.5)],
    )
    apply_inflows_to_domain(
        input_data=input_data, domain=mocks['domain'], start=start, duration=60,
        Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
        Inlet_operator=mocks['Inlet_operator'],
    )
    assert mocks['Polygonal_rate_operator'].call_count == 2
    assert mocks['Inlet_operator'].call_count == 2


def test_missing_keys_treated_as_empty(mocks, start):
    apply_inflows_to_domain(
        input_data={'boundary_polygon': BOUNDARY_POLYGON, 'catchment': {'features': []}},
        domain=mocks['domain'], start=start, duration=60,
        Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
        Inlet_operator=mocks['Inlet_operator'],
    )
    mocks['Polygonal_rate_operator'].assert_not_called()
    mocks['Inlet_operator'].assert_not_called()


def test_no_properties_type_filter_required(mocks, start):
    rainfall = _rainfall_feature()
    rainfall['properties'].pop('type', None)
    surface = _surface_feature()
    surface['properties'].pop('type', None)
    input_data = _input_data(rainfall_features=[rainfall], inflow_features=[surface])
    apply_inflows_to_domain(
        input_data=input_data, domain=mocks['domain'], start=start, duration=60,
        Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
        Inlet_operator=mocks['Inlet_operator'],
    )
    assert mocks['Polygonal_rate_operator'].call_count == 1
    assert mocks['Inlet_operator'].call_count == 1
