"""TASK-3662 (epic 3585): every one of the 14 culvert (entrance code, shape,
material) combinations maps to exactly one HDS-5 / SWMM culvert entry in the
engine, and an invalid combination is refused in plain words.

The table is the hydrata/gmc one (hydrata gn_anuga.models.culvert
CULVERT_ENTRANCE_TYPES, gmc culvertForm.js ENTRANCE_TYPES); the engine
catalogue is anuga_drainage.culverts.catalogue. Needs anuga_drainage (marker
requires_anuga).
"""
import pytest

pytestmark = pytest.mark.requires_anuga

#: (value, shape, material, hds5) - copied from hydrata CULVERT_ENTRANCE_TYPES.
ENTRANCE_TYPES = [
    ("square_edge_headwall", "circular", "concrete", "rcp_sq_hw"),
    ("groove_end_headwall", "circular", "concrete", "rcp_groove_hw"),
    ("groove_end_projecting", "circular", "concrete", "rcp_groove_proj"),
    ("bevelled_ring_45", "circular", "concrete", "circ_bevel45"),
    ("square_edge_headwall", "circular", "corrugated_metal", "cmp_hw"),
    ("mitred_to_slope", "circular", "corrugated_metal", "cmp_mitered"),
    ("projecting", "circular", "corrugated_metal", "cmp_proj"),
    ("square_edge_headwall", "box", "concrete", "box_sq_90hw"),
    ("wingwalls_30_75", "box", "concrete", "box_wing30_75"),
    ("wingwalls_0", "box", "concrete", "box_wing0"),
    ("wingwalls_45_top_bevel", "box", "concrete", "box_wing45_bev"),
    ("headwall_bevelled_45", "box", "concrete", "box_hw_bev45"),
    ("square_edge_headwall", "box", "corrugated_metal", "cmbox_90hw"),
    ("thin_wall_projecting", "box", "corrugated_metal", "cmbox_thinproj"),
]


def _spec(shape, material, entrance_type):
    spec = {
        "name": "Road", "shape": shape, "material": material, "entrance_type": entrance_type,
        "coordinates": [[0.0, 0.0], [20.0, 0.0]], "upstream_invert_m": 10.0, "downstream_invert_m": 9.9,
    }
    spec.update({"width_m": 1.5, "height_m": 1.2} if shape == "box" else {"diameter_m": 0.9})
    return spec


def _from_dict(spec):
    from anuga_drainage.culverts import HeadwallCulvert
    return HeadwallCulvert.from_dict(spec)


@pytest.mark.parametrize("value, shape, material, hds5", ENTRANCE_TYPES)
def test_culvert_entrance_each_combination_maps_to_its_hds5_entry(value, shape, material, hds5):
    entrance = _from_dict(_spec(shape, material, value)).entrance
    assert (entrance.value, entrance.shape, entrance.material, entrance.hds5) == (value, shape, material, hds5)


def test_culvert_entrance_14_combinations_map_to_14_distinct_swmm_entries():
    from anuga_drainage.culverts import ENTRANCE_TYPES as ENGINE
    assert len(ENGINE) == 14
    assert [(t.value, t.shape, t.material, t.hds5) for t in ENGINE] == ENTRANCE_TYPES
    assert len({t.swmm_code for t in ENGINE}) == 14
    assert len({t.hds5 for t in ENGINE}) == 14


def test_culvert_entrance_invalid_combination_is_refused_in_plain_words():
    with pytest.raises(ValueError, match="wingwalls_30_75.*not valid for a circular concrete"):
        _from_dict(_spec("circular", "concrete", "wingwalls_30_75"))
