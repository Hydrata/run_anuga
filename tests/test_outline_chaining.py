"""TASK-3457 (epic 3456): the domain outline is what the user drew.

``create_boundary_polygon_from_boundaries`` used to sort every External
vertex by polar angle around the bbox midpoint (commit 55db820). For an outline
that is not star-shaped from that point it built a DIFFERENT polygon (run 1574:
7.1 km² drawn, 9.4 km² built, the inflow fell outside and the run went dry).

These tests pin the replacement: External lines CHAINED end to end by a
deterministic rule (independent of feature order and drawn direction), oriented
clockwise BEFORE tags are assigned, with named errors for outlines that cannot
be built. Made-up shapes only (no customer geometry).
"""

import itertools

import pytest

pytest.importorskip("shapely", reason="shapely not installed")

from shapely.geometry import Polygon  # noqa: E402

from run_anuga import run_utils  # noqa: E402
from run_anuga.run_utils import (  # noqa: E402
    OUTLINE_DUPLICATE_ERROR,
    OUTLINE_GAP_ERROR,
    OUTLINE_INVALID_ERROR,
    FEATURE_OUTSIDE_ERROR,
    create_boundary_polygon_from_boundaries,
)

CRS = {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:EPSG::32616'}}
LEGACY_NO_EXTERNAL = (
    "create_boundary_polygon_from_boundaries: no valid External-location "
    "boundary coordinates found"
)


def _line(fid, coords, boundary):
    return {
        'type': 'Feature',
        'id': fid,
        'geometry': {'type': 'LineString', 'coordinates': coords},
        'properties': {'location': 'External', 'boundary': boundary},
    }


def _fc(*features, crs=True):
    fc = {'type': 'FeatureCollection', 'features': list(features)}
    if crs:
        fc['crs'] = CRS
    return fc


def _signed_area(ring):
    n = len(ring)
    return 0.5 * sum(ring[i][0] * ring[(i + 1) % n][1] - ring[(i + 1) % n][0] * ring[i][1]
                     for i in range(n))


def _assert_tags_cover_every_segment_once(ring, tags):
    indices = sorted(i for v in tags.values() for i in v)
    assert indices == list(range(len(ring)))


def _segment_tag(ring, tags, a, b):
    """Tag carried by the ring segment a -> b (either direction in the ring)."""
    n = len(ring)
    for i in range(n):
        seg = (tuple(ring[i]), tuple(ring[(i + 1) % n]))
        if seg in ((tuple(a), tuple(b)), (tuple(b), tuple(a))):
            return next(t for t, idx in tags.items() if i in idx)
    raise AssertionError(f"segment {a}->{b} not in ring {ring}")


# --------------------------------------------------------------------------- #
# Contract (A2 / AC4)                                                          #
# --------------------------------------------------------------------------- #
def test_error_prefix_constants_are_exported_verbatim():
    assert OUTLINE_GAP_ERROR == 'Boundary lines do not join up:'
    assert OUTLINE_INVALID_ERROR == 'Boundary outline is not a valid area:'
    assert OUTLINE_DUPLICATE_ERROR == 'Boundary lines conflict:'
    assert FEATURE_OUTSIDE_ERROR == 'Feature is outside the model area:'
    assert callable(run_utils.check_water_features_inside_outline)


def test_legacy_no_external_text_is_byte_identical():
    with pytest.raises(ValueError) as exc:
        create_boundary_polygon_from_boundaries(_fc())
    assert str(exc.value) == LEGACY_NO_EXTERNAL


# --------------------------------------------------------------------------- #
# A6-1: a single zig-zag line is the shape as drawn                            #
# --------------------------------------------------------------------------- #
ZIGZAG = [[0, 0], [100, 0], [100, 100], [60, 100], [60, 20], [40, 20], [40, 100], [0, 100]]


def test_zigzag_single_line_is_the_drawn_shape():
    ring, tags = create_boundary_polygon_from_boundaries(_fc(_line('z', ZIGZAG, 'Reflective')))
    built = Polygon(ring)
    assert built.area == pytest.approx(8400.0)
    assert built.symmetric_difference(Polygon(ZIGZAG)).area < 1e-9
    assert len(ring) == 8
    assert _signed_area(ring) < 0, "ring must be clockwise"
    assert tags == {'Reflective': list(range(8))}


# --------------------------------------------------------------------------- #
# A6-2: two-line square, one line drawn reversed: drawn tags, no zero-length   #
# --------------------------------------------------------------------------- #
SQUARE_D = [[0, 0], [100, 0], [100, 100]]
SQUARE_R_REVERSED = [[0, 0], [0, 100], [100, 100]]


def test_two_line_square_keeps_drawn_tags_and_has_no_zero_length_segment():
    ring, tags = create_boundary_polygon_from_boundaries(_fc(
        _line('d', SQUARE_D, 'Dirichlet'),
        _line('r', SQUARE_R_REVERSED, 'Reflective'),
    ))
    assert len(ring) == 4
    assert len({tuple(p) for p in ring}) == 4, "no zero-length segment"
    _assert_tags_cover_every_segment_once(ring, tags)
    # west + north are Reflective, east + south are Dirichlet
    assert _segment_tag(ring, tags, [0, 0], [0, 100]) == 'Reflective'
    assert _segment_tag(ring, tags, [0, 100], [100, 100]) == 'Reflective'
    assert _segment_tag(ring, tags, [100, 100], [100, 0]) == 'Dirichlet'
    assert _segment_tag(ring, tags, [100, 0], [0, 0]) == 'Dirichlet'
    assert _signed_area(ring) < 0


# --------------------------------------------------------------------------- #
# A6-3: exact duplicates dropped; a conflicting duplicate raises               #
# --------------------------------------------------------------------------- #
FOUR_SIDES = [
    ('s', [[0, 0], [100, 0]], 'Dirichlet'),
    ('e', [[100, 0], [100, 100]], 'Reflective'),
    ('n', [[100, 100], [0, 100]], 'Reflective'),
    ('w', [[0, 100], [0, 0]], 'Transmissive'),
]


def test_duplicate_lines_including_a_reversed_copy_are_dropped():
    feats = [_line(*f) for f in FOUR_SIDES]
    feats.append(_line('e-copy', [[100, 0], [100, 100]], 'Reflective'))
    feats.append(_line('n-reversed-copy', [[0, 100], [100, 100]], 'Reflective'))
    ring, tags = create_boundary_polygon_from_boundaries(_fc(*feats))
    assert len(ring) == 4
    assert Polygon(ring).area == pytest.approx(10000.0)
    _assert_tags_cover_every_segment_once(ring, tags)


def test_duplicate_line_with_a_different_boundary_type_raises():
    feats = [_line(*f) for f in FOUR_SIDES]
    feats.append(_line('w2', [[0, 0], [0, 100]], 'Dirichlet'))  # reversed copy of w
    with pytest.raises(ValueError) as exc:
        create_boundary_polygon_from_boundaries(_fc(*feats))
    msg = str(exc.value)
    assert msg.startswith(OUTLINE_DUPLICATE_ERROR)
    assert 'w and w2' in msg
    assert '(Dirichlet and Transmissive)' in msg


# --------------------------------------------------------------------------- #
# A6-4: only the closing edge may be open; other joins must be <= 1 m           #
# --------------------------------------------------------------------------- #
def test_two_gaps_raise_naming_both_features_and_the_small_gap():
    with pytest.raises(ValueError) as exc:
        create_boundary_polygon_from_boundaries(_fc(
            _line('a', [[0, 0], [100, 0], [100, 100]], 'Dirichlet'),
            _line('b', [[100, 130], [0, 340], [0, 240]], 'Reflective'),
        ))
    msg = str(exc.value)
    assert msg.startswith(OUTLINE_GAP_ERROR)
    assert '30.0 m' in msg
    assert 'boundary features a and b' in msg


def test_a_half_metre_join_is_merged():
    ring, tags = create_boundary_polygon_from_boundaries(_fc(
        _line('a', [[0, 0], [100, 0], [100, 100]], 'Dirichlet'),
        _line('b', [[100, 100.5], [0, 100], [0, 0]], 'Reflective'),
    ))
    assert len(ring) == 4
    assert Polygon(ring).area == pytest.approx(10000.0)
    _assert_tags_cover_every_segment_once(ring, tags)


# --------------------------------------------------------------------------- #
# A6-7: feature order and drawn direction never change the result              #
# --------------------------------------------------------------------------- #
def _reverse(feature):
    f = dict(feature)
    f['geometry'] = {'type': 'LineString', 'coordinates': feature['geometry']['coordinates'][::-1]}
    return f


RECTANGLE_3_SIDES = [  # stored [top, bottom, right]
    _line('top', [[0, 100], [100, 100]], 'Reflective'),
    _line('bottom', [[0, 0], [100, 0]], 'Dirichlet'),
    _line('right', [[100, 0], [100, 100]], 'Transmissive'),
]
L_TWO_LINES = [
    _line('x', [[0, 0], [100, 0]], 'Dirichlet'),
    _line('y', [[0, 0], [0, 100]], 'Reflective'),
]
TWO_LINE_SQUARE = [
    _line('d', SQUARE_D, 'Dirichlet'),
    _line('r', SQUARE_R_REVERSED, 'Reflective'),
]


@pytest.mark.parametrize('features, expected_ring, expected_tags', [
    pytest.param(
        RECTANGLE_3_SIDES,
        [[0, 100], [100, 100], [100, 0], [0, 0]],
        # closing edge = segment 3 (0,0)->(0,100), Dirichlet: the line it leaves
        {'Reflective': [0], 'Transmissive': [1], 'Dirichlet': [2, 3]},
        id='rectangle-3-sides',
    ),
    pytest.param(
        L_TWO_LINES,
        [[100, 0], [0, 0], [0, 100]],
        {'Dirichlet': [0], 'Reflective': [1, 2]},
        id='L-of-2-lines',
    ),
    pytest.param(
        TWO_LINE_SQUARE,
        [[0, 0], [0, 100], [100, 100], [100, 0]],
        {'Reflective': [0, 1], 'Dirichlet': [2, 3]},
        id='two-line-square',
    ),
])
def test_order_and_direction_independent(features, expected_ring, expected_tags):
    seen = set()
    n = len(features)
    for perm in itertools.permutations(features):
        for mask in range(2 ** n):
            feats = [_reverse(f) if mask >> i & 1 else f for i, f in enumerate(perm)]
            ring, tags = create_boundary_polygon_from_boundaries(_fc(*feats))
            assert ring == expected_ring, (perm, mask)
            assert tags == expected_tags, (perm, mask)
            seen.add(repr((ring, tags)))
    assert len(seen) == 1, "byte-identical across every order x reversal"


# --------------------------------------------------------------------------- #
# A6-8: an outline that is not a valid area raises                             #
# --------------------------------------------------------------------------- #
def test_bow_tie_raises():
    with pytest.raises(ValueError) as exc:
        create_boundary_polygon_from_boundaries(_fc(
            _line('bow', [[0, 0], [100, 100], [100, 0], [0, 100]], 'Reflective'),
        ))
    msg = str(exc.value)
    assert msg.startswith(OUTLINE_INVALID_ERROR)
    assert 'bow' in msg


def test_fewer_than_three_corners_raises():
    with pytest.raises(ValueError) as exc:
        create_boundary_polygon_from_boundaries(_fc(_line('seg', [[0, 0], [100, 0]], 'Reflective')))
    assert str(exc.value).startswith(OUTLINE_INVALID_ERROR)
    assert 'fewer than 3 distinct corners' in str(exc.value)


def test_feature_without_id_is_named_by_position():
    bow = _line(None, [[0, 0], [100, 100], [100, 0], [0, 100]], 'Reflective')
    bow.pop('id')
    with pytest.raises(ValueError, match='boundary feature #0'):
        create_boundary_polygon_from_boundaries(_fc(bow))


# --------------------------------------------------------------------------- #
# D8: crs is optional (it was vestigial)                                       #
# --------------------------------------------------------------------------- #
def test_crs_is_optional():
    feats = [_line(*f) for f in FOUR_SIDES]
    with_crs = create_boundary_polygon_from_boundaries(_fc(*feats))
    without_crs = create_boundary_polygon_from_boundaries(_fc(*feats, crs=False))
    assert with_crs == without_crs
    assert len(without_crs[0]) == 4


def test_internal_lines_and_multilinestring_parts():
    mls = {
        'type': 'Feature', 'id': 'south',
        'geometry': {'type': 'MultiLineString',
                     'coordinates': [[[0, 0], [50, 0]], [[50, 0], [100, 0]]]},
        'properties': {'location': 'External', 'boundary': 'Dirichlet'},
    }
    internal = {
        'type': 'Feature', 'id': 'wall',
        'geometry': {'type': 'LineString', 'coordinates': [[10, 10], [20, 20]]},
        'properties': {'location': 'Internal', 'boundary': 'Reflective'},
    }
    ring, tags = create_boundary_polygon_from_boundaries(_fc(
        mls, internal, _line('rest', [[100, 0], [100, 100], [0, 100], [0, 0]], 'Reflective'),
    ))
    assert Polygon(ring).area == pytest.approx(10000.0)
    assert len(ring) == 5  # the (50, 0) mid-vertex of the south line is kept
    assert set(tags) == {'Dirichlet', 'Reflective'}
    assert len(tags['Dirichlet']) == 2
