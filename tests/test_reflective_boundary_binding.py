"""TASK-3590 — a Reflective building's hole tag binds to a boundary condition.

make_interior_holes_and_tags tags every Reflective building hole 'reflective'
(lowercase); run_sim's boundary map only knew the capitalised 'Reflective', so
every run with a Reflective building died at set_boundary with
KeyError: 'reflective'. These tests pin the producer's tags against the map
run_sim actually binds.
"""

import os
import tempfile

import pytest

from run_anuga.run import bind_boundary_tags, make_default_boundary_maps
from run_anuga.run_utils import make_interior_holes_and_tags

BOUNDARY_POLYGON = [
    [321000.0, 5812000.0],
    [321200.0, 5812000.0],
    [321200.0, 5812200.0],
    [321000.0, 5812200.0],
]
BUILDING = [
    [321040.0, 5812080.0],
    [321060.0, 5812080.0],
    [321060.0, 5812100.0],
    [321040.0, 5812100.0],
    [321040.0, 5812080.0],
]


def _building_input(method='Reflective'):
    return {
        'building': {
            'type': 'FeatureCollection',
            'features': [{
                'type': 'Feature',
                'id': 'str_1',
                'geometry': {'type': 'Polygon', 'coordinates': [BUILDING]},
                'properties': {'method': method},
            }],
        }
    }


class _FakeAnuga:
    """Stands in for the anuga module: each factory returns its own name."""

    def Dirichlet_boundary(self, values):
        return 'Dirichlet_boundary'

    def Reflective_boundary(self, domain):
        return 'Reflective_boundary'

    def Transmissive_boundary(self, domain):
        return 'Transmissive_boundary'


class TestBoundaryMapCoversHoleTags:

    @pytest.mark.requires_geo
    def test_every_reflective_hole_tag_binds_to_reflective_boundary(self):
        _, hole_tags = make_interior_holes_and_tags(_building_input())
        tags = [tag for hole in hole_tags for tag in hole]
        assert tags, 'expected a tagged hole for a Reflective building'

        bound = bind_boundary_tags(tags, make_default_boundary_maps(_FakeAnuga(), domain=None))

        assert set(bound.values()) == {'Reflective_boundary'}

    def test_unknown_tag_names_the_tag(self):
        maps = make_default_boundary_maps(_FakeAnuga(), domain=None)
        with pytest.raises(ValueError, match="'culvert'"):
            bind_boundary_tags(['exterior', 'culvert'], maps)

    def test_external_tags_still_bind(self):
        maps = make_default_boundary_maps(_FakeAnuga(), domain=None)
        bound = bind_boundary_tags(
            ['exterior', 'Dirichlet', 'Reflective', 'Transmissive', 'interior'], maps
        )
        assert bound['Dirichlet'] == 'Dirichlet_boundary'
        assert bound['Reflective'] == 'Reflective_boundary'
        assert bound['Transmissive'] == 'Transmissive_boundary'


@pytest.mark.requires_anuga
class TestReflectiveBuildingDomainSetBoundary:
    """End to end: mesh with a Reflective hole -> Domain -> set_boundary succeeds."""

    def test_set_boundary_with_reflective_hole(self):
        import anuga
        from run_anuga.run_utils import create_anuga_mesh

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_data = {
                'mesh_filepath': os.path.join(tmp_dir, 'run_1_1_1.msh'),
                'scenario_config': {
                    'epsg': 'EPSG:28355', 'resolution': 20,
                    'project': 1, 'id': 1, 'run_id': 1,
                },
                'boundary_polygon': BOUNDARY_POLYGON,
                'boundary_tags': {'Dirichlet': list(range(len(BOUNDARY_POLYGON)))},
                **_building_input(),
            }
            msh_path, _ = create_anuga_mesh(input_data)
            domain = anuga.Domain(msh_path)

            tags = set(domain.boundary.values())
            assert 'reflective' in tags

            domain.set_boundary(
                bind_boundary_tags(domain.boundary.values(), make_default_boundary_maps(anuga, domain))
            )
            assert isinstance(domain.boundary_map['reflective'], anuga.Reflective_boundary)
