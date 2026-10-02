"""TASK-3186 (epic 3200, Option A) — MeshRegion ``resolution`` is a LENGTH.

A mesh region's ``resolution`` is a target edge length in metres, exactly like
``scenario.resolution`` (and exactly what the UI has promised since gmc
b9892e052). ANUGA's ``interior_regions`` take a maximum triangle AREA, so the
conversion ``res**2 / 2`` happens INSIDE ``make_interior_regions`` — never at
the consumer, because ``make_breaklines`` already emits areas into the same
list and converting there would square every breakline ring.

Because stored data used to hold AREAS (the Towradgi bundle import), a region
feature with a resolution must carry the units marker ``resolution_units ==
'm'`` (stamped by hydrata's ``convert_mesh_region_units`` command, or by the
column default on layers created after the fix). An unmarked one is REFUSED
loudly, naming the project and the command — never meshed 6-9x too coarse.
"""

import math
import os
import tempfile

import pytest

from run_anuga.run_utils import (
    MESH_REGION_UNITS_KEY,
    MESH_REGION_UNITS_LENGTH_M,
    MeshRegionUnitsUnmarked,
    make_interior_regions,
)


def _square(x0, y0, size):
    return [[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size], [x0, y0]]


def _region(resolution, marker=MESH_REGION_UNITS_LENGTH_M, fid='r1', key=MESH_REGION_UNITS_KEY):
    props = {'resolution': resolution}
    if marker is not None:
        props[key] = marker
    return {'type': 'Feature', 'id': fid,
            'geometry': {'type': 'Polygon', 'coordinates': [_square(321000, 5812000, 50)]},
            'properties': props}


@pytest.mark.requires_geo
class TestMeshRegionUnitConversion:
    def test_mesh_region_unit_marked_length_becomes_area(self):
        regions = make_interior_regions({'mesh_region': {'features': [_region(10.0)]}})
        assert regions[0][1] == pytest.approx(50.0)  # 10 m edge -> 10**2/2 m2

    def test_mesh_region_unit_two_regions_each_converted(self):
        regions = make_interior_regions({'mesh_region': {'features': [
            _region(4.0, fid='a'), _region(14.142, fid='b')]}})
        assert [r[1] for r in regions] == [pytest.approx(8.0), pytest.approx(100.0, rel=1e-3)]

    def test_mesh_region_unit_marker_key_case_insensitive(self):
        regions = make_interior_regions({'mesh_region': {'features': [
            _region(6.0, key='Resolution_Units', marker='M')]}})
        assert regions[0][1] == pytest.approx(18.0)

    def test_mesh_region_unit_null_resolution_needs_no_marker(self):
        regions = make_interior_regions({'mesh_region': {'features': [_region(None, marker=None)]}})
        assert regions[0][1] is None

    def test_mesh_region_unit_unmarked_region_is_refused(self):
        input_data = {'scenario_config': {'project': 850, 'id': 427},
                      'mesh_region': {'features': [_region(8.0, marker=None, fid='mes.3')]}}
        with pytest.raises(MeshRegionUnitsUnmarked) as exc:
            make_interior_regions(input_data)
        msg = str(exc.value)
        assert 'project 850' in msg
        assert 'mes.3' in msg
        assert 'convert_mesh_region_units --project 850' in msg

    def test_mesh_region_unit_wrong_marker_value_is_refused(self):
        with pytest.raises(MeshRegionUnitsUnmarked):
            make_interior_regions({'mesh_region': {'features': [_region(8.0, marker='m2')]}})

    def test_mesh_region_unit_refusal_is_a_value_error(self):
        # Build error paths record str(error) verbatim (services.build_simulation_package).
        assert issubclass(MeshRegionUnitsUnmarked, ValueError)


@pytest.mark.requires_anuga
class TestMeshRegionUnitRealMesh:
    """The real mesher: a marked LENGTH region meshes at res**2/2, not at res."""

    def _count(self, region_resolution):
        pytest.importorskip("anuga", reason="anuga not installed")
        pytest.importorskip("meshpy.triangle", reason="meshpy not installed")
        from run_anuga.run_utils import create_anuga_mesh
        with tempfile.TemporaryDirectory() as tmp:
            input_data = {
                'mesh_filepath': os.path.join(tmp, 'run_1_1_1.msh'),
                'scenario_config': {'epsg': 'EPSG:28355', 'resolution': 40, 'project': 1, 'id': 1, 'run_id': 1},
                'boundary_polygon': [[321000.0, 5812000.0], [322000.0, 5812000.0],
                                     [322000.0, 5813000.0], [321000.0, 5813000.0]],
                'boundary_tags': {'exterior': [0, 1, 2, 3]},
                'mesh_region': {'features': [{
                    'type': 'Feature', 'id': 'r',
                    'geometry': {'type': 'Polygon', 'coordinates': [_square(321350, 5812350, 300)]},
                    'properties': {'resolution': region_resolution, MESH_REGION_UNITS_KEY: 'm'}}]},
            }
            _, mesh = create_anuga_mesh(input_data)
            return len(mesh.tri_mesh.triangles)

    def test_mesh_region_unit_real_mesh_density_matches_length(self):
        n = self._count(5.0)
        # region 300x300 m at 5 m edge -> ideal 90000 / 12.5 = 7200 triangles;
        # Triangle's quality constraint yields ~1.5x the ideal packing. Base:
        # (1e6 - 9e4) / 800 ~ 1138 ideal. The AREA misreading (cap 5 m2) would
        # give ~18000 ideal in the region alone (> 27000 meshed).
        ideal = 90000 / (5.0 ** 2 / 2) + (1e6 - 9e4) / (40 ** 2 / 2)
        assert 0.9 * ideal <= n <= 2.2 * ideal, (n, ideal)
        assert n < 90000 / 5.0, n  # NOT the area misreading
        assert math.isfinite(n)
