"""Consumer-site shape-variance integration tests for run_utils.py geometry handling."""

import importlib.util
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


# Module-top dependency gates. shapely + pandas live in the [dev] extras so
# any environment running the test suite has them. Failing fast here yields
# a SKIPPED module (rather than a collection error) on a leaner install.
pytest.importorskip('shapely')
pytest.importorskip('pandas')

_HAS_OSGEO = importlib.util.find_spec('osgeo') is not None
_requires_osgeo = pytest.mark.skipif(not _HAS_OSGEO, reason='osgeo required')


# Geometry fixtures shared across classes. Kept module-scope so individual
# tests stay short.

OUTER_RING = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]
OUTER_RING_2 = [[50.0, 50.0], [60.0, 50.0], [60.0, 60.0], [50.0, 60.0], [50.0, 50.0]]
LINE_COORDS_A = [[15.0, 15.0], [25.0, 15.0]]
LINE_COORDS_B = [[30.0, 15.0], [40.0, 15.0]]


def _polygon_feature(fid='f.1', ring=None, properties=None):
    return {
        'type': 'Feature',
        'id': fid,
        'geometry': {'type': 'Polygon', 'coordinates': [ring or OUTER_RING]},
        'properties': properties or {},
    }


def _multipolygon_feature(fid='f.1', rings=None, properties=None):
    rings = rings or [OUTER_RING]
    return {
        'type': 'Feature',
        'id': fid,
        'geometry': {'type': 'MultiPolygon', 'coordinates': [[r] for r in rings]},
        'properties': properties or {},
    }


def _empty_polygon_feature(fid='f.empty', kind='Polygon', properties=None):
    return {
        'type': 'Feature',
        'id': fid,
        'geometry': {'type': kind, 'coordinates': []},
        'properties': properties or {},
    }


# --------------------------------------------------------------------------- #
# Polygon-consumer empty/None gaps (audit table rows: make_frictions,         #
# make_interior_regions, make_interior_holes_and_tags × {empty, None})        #
# --------------------------------------------------------------------------- #


class TestMakeFrictionsEmptyAndNone:
    """Audit gap: ``make_frictions`` × {empty, None} (both Polygon + MultiPolygon).

    The hardened ``_extract_polygon_outer_ring`` returns ``[]`` for empty /
    missing coordinates, so the consumer should produce an empty-ring friction
    tuple rather than crash. The default ``All`` row is always appended.
    """

    def test_friction_polygon_empty_coordinates_yields_empty_ring(self):
        from run_anuga.run_utils import make_frictions
        input_data = {'friction': {'features': [
            _empty_polygon_feature(kind='Polygon', properties={'mannings': 0.03}),
        ]}}
        result = make_frictions(input_data)
        assert result[0] == ([], 0.03)

    def test_friction_multipolygon_empty_coordinates_yields_empty_ring(self):
        from run_anuga.run_utils import make_frictions
        input_data = {'friction': {'features': [
            _empty_polygon_feature(kind='MultiPolygon', properties={'mannings': 0.03}),
        ]}}
        result = make_frictions(input_data)
        assert result[0] == ([], 0.03)

    def test_friction_polygon_none_coordinates_yields_empty_ring(self):
        from run_anuga.run_utils import make_frictions
        input_data = {'friction': {'features': [{
            'type': 'Feature',
            'geometry': {'type': 'Polygon', 'coordinates': None},
            'properties': {'mannings': 0.04},
        }]}}
        result = make_frictions(input_data)
        assert result[0] == ([], 0.04)

    def test_mannings_structure_empty_polygon_yields_empty_ring(self):
        from run_anuga.run_utils import make_frictions
        input_data = {'structure': {'features': [
            _empty_polygon_feature(kind='Polygon', properties={'method': 'Mannings'}),
        ]}}
        result = make_frictions(input_data)
        # First entry is the empty Mannings structure polygon
        assert result[0][0] == []

    def test_friction_no_features_returns_only_default_all(self):
        from run_anuga.run_utils import make_frictions
        from run_anuga import defaults
        result = make_frictions({})
        assert result == [['All', defaults.DEFAULT_MANNINGS_N]]


class TestMakeInteriorRegionsEmptyAndNone:
    """Audit gap: ``make_interior_regions`` × {empty, None}."""

    def test_polygon_empty_coordinates_yields_empty_ring_tuple(self):
        from run_anuga.run_utils import make_interior_regions
        input_data = {'mesh_region': {'features': [
            _empty_polygon_feature(kind='Polygon', properties={'resolution': 5.0}),
        ]}}
        assert make_interior_regions(input_data) == [([], 5.0)]

    def test_multipolygon_empty_coordinates_yields_empty_ring_tuple(self):
        from run_anuga.run_utils import make_interior_regions
        input_data = {'mesh_region': {'features': [
            _empty_polygon_feature(kind='MultiPolygon', properties={'resolution': 5.0}),
        ]}}
        assert make_interior_regions(input_data) == [([], 5.0)]

    def test_polygon_none_coordinates_yields_empty_ring_tuple(self):
        from run_anuga.run_utils import make_interior_regions
        input_data = {'mesh_region': {'features': [{
            'type': 'Feature',
            'geometry': {'type': 'Polygon', 'coordinates': None},
            'properties': {'resolution': 5.0},
        }]}}
        assert make_interior_regions(input_data) == [([], 5.0)]

    def test_polygon_valid_real_coordinates_yields_ring(self):
        from run_anuga.run_utils import make_interior_regions
        input_data = {'mesh_region': {'features': [
            _polygon_feature(properties={'resolution': 5.0}),
        ]}}
        assert make_interior_regions(input_data) == [(OUTER_RING, 5.0)]


class TestMakeInteriorHolesEmptyAndNone:
    """Audit gap: ``make_interior_holes_and_tags`` × {empty, None}."""

    def test_polygon_empty_coordinates_yields_empty_ring(self):
        from run_anuga.run_utils import make_interior_holes_and_tags
        input_data = {'structure': {'features': [
            _empty_polygon_feature(kind='Polygon', properties={'method': 'Holes'}),
        ]}}
        holes, tags = make_interior_holes_and_tags(input_data)
        assert holes == [[]]
        assert tags == [None]

    def test_multipolygon_empty_coordinates_yields_empty_ring(self):
        from run_anuga.run_utils import make_interior_holes_and_tags
        input_data = {'structure': {'features': [
            _empty_polygon_feature(kind='MultiPolygon', properties={'method': 'Holes'}),
        ]}}
        holes, tags = make_interior_holes_and_tags(input_data)
        assert holes == [[]]
        assert tags == [None]

    def test_polygon_reflective_method_yields_reflective_tag(self):
        from run_anuga.run_utils import make_interior_holes_and_tags
        input_data = {'structure': {'features': [
            _polygon_feature(properties={'method': 'Reflective'}),
        ]}}
        holes, tags = make_interior_holes_and_tags(input_data)
        assert holes == [OUTER_RING]
        assert tags == [{'reflective': list(range(len(OUTER_RING)))}]

    def test_multipolygon_reflective_method_yields_reflective_tag(self):
        from run_anuga.run_utils import make_interior_holes_and_tags
        input_data = {'structure': {'features': [
            _multipolygon_feature(properties={'method': 'Reflective'}),
        ]}}
        holes, tags = make_interior_holes_and_tags(input_data)
        assert holes == [OUTER_RING]
        assert tags == [{'reflective': list(range(len(OUTER_RING)))}]


# --------------------------------------------------------------------------- #
# apply_inflows_to_domain × Polygon/MultiPolygon (rainfall + catchment)        #
# (audit table rows L885 + L924, needs shapely for                            #
# check_coordinates_are_in_polygon, needs pandas for inflow_dataframe.)        #
# --------------------------------------------------------------------------- #


class TestApplyInflowsRainfallPolygonShape:
    """Audit gap: ``apply_inflows_to_domain`` rainfall polygon × {Polygon, MultiPolygon}.

    Asserts the ``polygon=`` kwarg passed to ``Polygonal_rate_operator`` is
    the 2-D outer ring regardless of whether the source feature was
    ``Polygon`` or ``MultiPolygon``.
    """

    BOUNDARY = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]

    def _input_data(self, rainfall_feature):
        return {
            'rainfall': {'features': [rainfall_feature]},
            'inflow': {'features': []},
            'catchment': {'features': []},
            'boundary_polygon': self.BOUNDARY,
        }

    def _call(self, rainfall_feature):
        from run_anuga.run_utils import apply_inflows_to_domain
        domain = MagicMock(name='domain')
        prate = MagicMock(name='Polygonal_rate_operator')
        inlet = MagicMock(name='Inlet_operator')
        apply_inflows_to_domain(
            input_data=self._input_data(rainfall_feature),
            domain=domain,
            start=datetime(2020, 1, 1, tzinfo=timezone.utc),
            duration=60,
            Polygonal_rate_operator=prate,
            Inlet_operator=inlet,
        )
        return prate

    def test_polygon_rainfall_passes_outer_ring_to_polygonal_rate_operator(self):
        rainfall = _polygon_feature(fid='r.1', properties={'data': 5.0})
        prate = self._call(rainfall)
        assert prate.call_count == 1
        assert prate.call_args.kwargs['polygon'] == OUTER_RING

    def test_multipolygon_rainfall_passes_outer_ring_to_polygonal_rate_operator(self):
        rainfall = _multipolygon_feature(fid='r.1', properties={'data': 5.0})
        prate = self._call(rainfall)
        assert prate.call_count == 1
        assert prate.call_args.kwargs['polygon'] == OUTER_RING


CATCHMENT_INNER_RING = [
    [20.0, 20.0], [40.0, 20.0], [40.0, 40.0], [20.0, 40.0], [20.0, 20.0],
]


class TestApplyInflowsCatchmentPolygonShape:
    """Audit gap: ``apply_inflows_to_domain`` catchment polygon × {Polygon, MultiPolygon}.

    Catchment is only consulted when a rainfall feature with a constant
    ``data`` value is also present, so we supply one. The catchment ring is
    strictly inside the boundary polygon (shapely.Polygon.contains is strict,
    so boundary-touching rings get rejected), so the inner
    ``check_coordinates_are_in_polygon`` test succeeds and a *second*
    ``Polygonal_rate_operator`` invocation fires with the catchment ring as
    ``polygon=``.
    """

    BOUNDARY = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]

    def _input_data(self, catchment_feature):
        return {
            'rainfall': {'features': [
                _polygon_feature(fid='r.1', properties={'data': 5.0}),
            ]},
            'inflow': {'features': []},
            'catchment': {'features': [catchment_feature]},
            'boundary_polygon': self.BOUNDARY,
        }

    def _call(self, catchment_feature):
        from run_anuga.run_utils import apply_inflows_to_domain
        prate = MagicMock(name='Polygonal_rate_operator')
        apply_inflows_to_domain(
            input_data=self._input_data(catchment_feature),
            domain=MagicMock(name='domain'),
            start=datetime(2020, 1, 1, tzinfo=timezone.utc),
            duration=60,
            Polygonal_rate_operator=prate,
            Inlet_operator=MagicMock(name='Inlet_operator'),
        )
        return prate

    def test_polygon_catchment_passes_outer_ring_to_polygonal_rate_operator(self):
        catchment = _polygon_feature(fid='c.1', ring=CATCHMENT_INNER_RING)
        prate = self._call(catchment)
        # call[0] = rainfall, call[1] = catchment
        assert prate.call_count == 2
        assert prate.call_args_list[1].kwargs['polygon'] == CATCHMENT_INNER_RING

    def test_multipolygon_catchment_passes_outer_ring_to_polygonal_rate_operator(self):
        catchment = _multipolygon_feature(fid='c.1', rings=[CATCHMENT_INNER_RING])
        prate = self._call(catchment)
        assert prate.call_count == 2
        assert prate.call_args_list[1].kwargs['polygon'] == CATCHMENT_INNER_RING


class TestApplyInflowsInflowLineShape:
    """Audit gap: ``apply_inflows_to_domain`` inflow line × {LineString,
    MultiLineString >1 sub-line}.

    A MultiLineString with more than one sub-line is the historically-untested
    real-multi path. The flattener concatenates sub-lines, so the resulting
    ``Inlet_operator`` should receive the concatenated coords. The line lives
    inside the boundary polygon so the ``check_coordinates_are_in_polygon``
    gate passes.
    """

    BOUNDARY = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]

    def _call(self, inflow_feature):
        from run_anuga.run_utils import apply_inflows_to_domain
        inlet = MagicMock(name='Inlet_operator')
        apply_inflows_to_domain(
            input_data={
                'rainfall': {'features': []},
                'inflow': {'features': [inflow_feature]},
                'catchment': {'features': []},
                'boundary_polygon': self.BOUNDARY,
            },
            domain=MagicMock(name='domain'),
            start=datetime(2020, 1, 1, tzinfo=timezone.utc),
            duration=60,
            Polygonal_rate_operator=MagicMock(name='Polygonal_rate_operator'),
            Inlet_operator=inlet,
        )
        return inlet

    def test_linestring_inflow_passes_coords_to_inlet_operator(self):
        feature = {
            'type': 'Feature',
            'id': 's.1',
            'geometry': {'type': 'LineString', 'coordinates': LINE_COORDS_A},
            'properties': {'data': 1.5},
        }
        inlet = self._call(feature)
        assert inlet.call_count == 1
        assert inlet.call_args.args[1] == LINE_COORDS_A

    def test_multilinestring_one_subline_inflow_passes_flattened_coords(self):
        feature = {
            'type': 'Feature',
            'id': 's.1',
            'geometry': {'type': 'MultiLineString', 'coordinates': [LINE_COORDS_A]},
            'properties': {'data': 1.5},
        }
        inlet = self._call(feature)
        assert inlet.call_count == 1
        assert inlet.call_args.args[1] == LINE_COORDS_A

    def test_multilinestring_multiple_sublines_inflow_passes_concatenated_coords(self):
        feature = {
            'type': 'Feature',
            'id': 's.1',
            'geometry': {
                'type': 'MultiLineString',
                'coordinates': [LINE_COORDS_A, LINE_COORDS_B],
            },
            'properties': {'data': 1.5},
        }
        inlet = self._call(feature)
        assert inlet.call_count == 1
        assert inlet.call_args.args[1] == LINE_COORDS_A + LINE_COORDS_B


# --------------------------------------------------------------------------- #
# calculate_hydrology × Polygon/MultiPolygon                                  #
# (audit table rows L1252 + L1273).                                            #
# --------------------------------------------------------------------------- #


class TestCalculateHydrologyCatchmentContainment:
    """Audit gap: ``calculate_hydrology`` catchment containment × {Polygon, MultiPolygon}.

    Stubs ``setup_input_data`` so the test doesn't need a real package_dir.
    Verifies the assigned ``node_id`` reflects the catchment's polygon
    containment of the node, regardless of geometry shape.
    """

    def _input_data(self, catchment_feature):
        # Node lives at the center of OUTER_RING (5, 5), inside the catchment.
        return {
            'nodes': {'features': [{
                'id': 'node-A',
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [5.0, 5.0]},
                'properties': {},
            }]},
            'catchment': {'features': [catchment_feature]},
            'rainfall': {'features': []},
            'inflow_filename': '/tmp/inflow.geojson',
        }

    def _call(self, catchment_feature):
        from run_anuga import run_utils
        input_data = self._input_data(catchment_feature)
        with patch.object(run_utils, 'setup_input_data', return_value=input_data):
            run_utils.calculate_hydrology('/tmp/nonexistent_pkg')
        return input_data

    def test_polygon_catchment_assigns_contained_node_id(self):
        catchment = _polygon_feature(fid='c.1', properties={})
        result = self._call(catchment)
        assert result['catchment']['features'][0].get('node_id') == 'node-A'

    def test_multipolygon_catchment_assigns_contained_node_id(self):
        catchment = _multipolygon_feature(fid='c.1', properties={})
        result = self._call(catchment)
        assert result['catchment']['features'][0].get('node_id') == 'node-A'


class TestCalculateHydrologyAreaM2:
    """Audit gap: ``calculate_hydrology`` area_m2 × {Polygon, MultiPolygon}.

    The OUTER_RING is a 10x10 square = 100 m². The catchment area is
    multiplied by the rainfall steady-state intensity to yield
    ``surface_flow_m3_s``. We verify the surface flow value matches
    ``intensity_m_s * 100`` for both shapes.
    """

    def _input_data(self, catchment_feature, rainfall_mm_hr=3.6):
        # Place node at center so the catchment-->node assignment works.
        return {
            'nodes': {'features': [{
                'id': 'node-A',
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [5.0, 5.0]},
                'properties': {},
            }]},
            'catchment': {'features': [catchment_feature]},
            'rainfall': {'features': [_polygon_feature(
                fid='r.1', properties={'data': rainfall_mm_hr},
            )]},
            'inflow_filename': '/tmp/inflow.geojson',
        }

    def _call(self, catchment_feature):
        # add_inflow_to_file would try to open a real file; patch it out.
        from run_anuga import run_utils
        input_data = self._input_data(catchment_feature)
        with patch.object(run_utils, 'setup_input_data', return_value=input_data), \
             patch.object(run_utils, 'add_inflow_to_file'):
            run_utils.calculate_hydrology('/tmp/nonexistent_pkg')
        return input_data

    def test_polygon_catchment_area_yields_expected_surface_flow(self):
        catchment = _polygon_feature(fid='c.1', properties={})
        result = self._call(catchment)
        # 3.6 mm/hr -> 1e-6 m/s; area = 100 m² -> surface_flow_m3_s = 1e-4
        assigned = result['catchment']['features'][0].get('surface_flow_m3_s')
        assert assigned == pytest.approx(100.0 * (3.6 * 0.001 / 3600))

    def test_multipolygon_catchment_area_yields_expected_surface_flow(self):
        catchment = _multipolygon_feature(fid='c.1', properties={})
        result = self._call(catchment)
        assigned = result['catchment']['features'][0].get('surface_flow_m3_s')
        assert assigned == pytest.approx(100.0 * (3.6 * 0.001 / 3600))


# --------------------------------------------------------------------------- #
# create_mesher_mesh × Polygon/MultiPolygon (audit row L182)                  #
# Heavy function, short-circuit by raising from make_shp_from_polygon.       #
# --------------------------------------------------------------------------- #


@_requires_osgeo
class TestCreateMesherMeshPolygonShape:
    """Audit gap: ``create_mesher_mesh`` × {Polygon, MultiPolygon}.

    The function walks ``input_data['mesh_region']['features']``, builds
    a shapefile via ``make_shp_from_polygon`` (the L182 site), and then
    calls a long chain of ``gdalwarp`` / mesher subprocesses. We short-
    circuit by raising from a patched ``make_shp_from_polygon`` once we've
    captured its first-call args; downstream subprocesses never run.

    Asserts ``make_shp_from_polygon`` receives a flat 2-D outer ring as
    its first argument regardless of whether the source mesh_region
    feature was Polygon or MultiPolygon.
    """

    class _ShortCircuit(Exception):
        pass

    def _input_data(self, mesh_region_feature, tmp_path):
        return {
            'output_directory': str(tmp_path),
            'elevation_filename': str(tmp_path / 'elev.tif'),
            'mesh_region_filename': str(tmp_path / 'mesh_region.geojson'),
            'mesh_region': {
                'crs': {
                    'type': 'name',
                    'properties': {'name': 'urn:ogc:def:crs:EPSG::32616'},
                },
                'features': [mesh_region_feature],
            },
            'resolution': 5.0,
            'scenario_config': {'max_rmse_tolerance': 1},
        }

    def _call_and_capture(self, mesh_region_feature, tmp_path):
        from run_anuga import run_utils
        captured = {}

        def fake_make_shp(ring, epsg, path, buffer=0):
            captured['ring'] = ring
            captured['epsg'] = epsg
            raise TestCreateMesherMeshPolygonShape._ShortCircuit('captured')

        # gdal.Open needs an iterable GetGeoTransform; mock it.
        fake_raster = MagicMock()
        fake_raster.GetGeoTransform.return_value = (0, 5.0, 0, 0, 0, -5.0)
        with patch.object(run_utils, 'make_shp_from_polygon', side_effect=fake_make_shp), \
             patch('run_anuga.run_utils.import_optional') as imp_opt:
            # gdal first, ogr second; order matches the function source.
            fake_gdal = MagicMock()
            fake_gdal.Open.return_value = fake_raster
            fake_ogr = MagicMock()
            imp_opt.side_effect = lambda name: fake_gdal if name == 'osgeo.gdal' else fake_ogr
            with pytest.raises(TestCreateMesherMeshPolygonShape._ShortCircuit):
                run_utils.create_mesher_mesh(self._input_data(mesh_region_feature, tmp_path))
        return captured

    def test_polygon_mesh_region_passes_outer_ring_to_make_shp(self, tmp_path):
        feature = _polygon_feature(fid='mr.1', properties={'resolution': 5.0})
        captured = self._call_and_capture(feature, tmp_path)
        assert captured['ring'] == OUTER_RING
        assert captured['epsg'] == 32616

    def test_multipolygon_mesh_region_passes_outer_ring_to_make_shp(self, tmp_path):
        feature = _multipolygon_feature(fid='mr.1', properties={'resolution': 5.0})
        captured = self._call_and_capture(feature, tmp_path)
        assert captured['ring'] == OUTER_RING
        assert captured['epsg'] == 32616
