"""
TASK-1259: Friction-raster precedence merges raster + per-structure patches.
TASK-1260: Robust CRS/UTM-zone derivation via pyproj.
"""
import pytest


# --------------------------------------------------------------------------- #
# TASK-1259: make_frictions with raster + structure Manning's overlay          #
# --------------------------------------------------------------------------- #

OUTER_RING = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]
OUTER_RING_2 = [[50.0, 50.0], [60.0, 50.0], [60.0, 60.0], [50.0, 60.0], [50.0, 50.0]]


def _polygon_feature(ring=None, properties=None):
    return {
        'type': 'Feature',
        'geometry': {'type': 'Polygon', 'coordinates': [ring or OUTER_RING]},
        'properties': properties or {},
    }


class TestMakeFrictionsRasterPlusStructure:
    """
    TASK-1259: When a friction_raster_filename is present AND structures with
    Manning's-n patches exist, make_frictions must return BOTH:
      - the raster as ['Extent', path] (first, lowest priority in ANUGA's
        composite_quantity_setting_function which processes last-wins, but
        the raster is domain-wide background)
      - each structure polygon's Manning's-n as a patch ON TOP of the raster

    The old early-return dropped the structure patches entirely.
    """

    def _input_data_raster_only(self, raster_path='/tmp/friction.tif'):
        return {
            'friction_raster_filename': raster_path,
        }

    def _input_data_raster_plus_structure(self, raster_path='/tmp/friction.tif'):
        return {
            'friction_raster_filename': raster_path,
            'structure': {
                'features': [
                    _polygon_feature(ring=OUTER_RING, properties={'method': 'Mannings'}),
                    _polygon_feature(ring=OUTER_RING_2, properties={'method': 'Reflective'}),
                ]
            },
        }

    def test_raster_only_returns_extent_entry(self):
        """No structure: just the Extent entry (unchanged behaviour)."""
        from run_anuga.run_utils import make_frictions
        result = make_frictions(self._input_data_raster_only('/some/raster.tif'))
        assert result[0] == ['Extent', '/some/raster.tif']

    def test_raster_with_mannings_structure_includes_both(self):
        """
        With raster + 1 Mannings structure + 1 Reflective structure:
        result must contain the raster entry AND the Mannings patch,
        but NOT the Reflective structure (non-Manning method).
        """
        from run_anuga.run_utils import make_frictions
        result = make_frictions(self._input_data_raster_plus_structure('/path/friction.tif'))

        # Should have: [Extent entry, Mannings structure patch]
        # The All default should NOT be appended when a raster is present
        # (the raster already covers the whole domain).
        raster_entries = [e for e in result if isinstance(e, list) and e[0] == 'Extent']
        assert len(raster_entries) == 1, f"Expected 1 Extent entry, got: {result}"
        assert raster_entries[0][1] == '/path/friction.tif'

        # Mannings structure patch must also be present
        patch_entries = [e for e in result if isinstance(e, tuple)]
        assert len(patch_entries) == 1, f"Expected 1 structure patch, got: {result}"
        assert patch_entries[0][0] == OUTER_RING  # the Mannings polygon ring
        from run_anuga import defaults
        assert patch_entries[0][1] == defaults.BUILDING_MANNINGS_N

    def test_raster_with_no_mannings_structure_returns_only_raster(self):
        """Structure present but all Reflective (non-Mannings): just raster."""
        from run_anuga.run_utils import make_frictions
        input_data = {
            'friction_raster_filename': '/raster.tif',
            'structure': {
                'features': [
                    _polygon_feature(properties={'method': 'Reflective'}),
                ]
            },
        }
        result = make_frictions(input_data)
        assert len(result) == 1
        assert result[0] == ['Extent', '/raster.tif']

    def test_no_raster_with_structure_unchanged(self):
        """Without raster: existing behaviour (structure patches + All default)."""
        from run_anuga.run_utils import make_frictions
        input_data = {
            'structure': {
                'features': [
                    _polygon_feature(ring=OUTER_RING, properties={'method': 'Mannings'}),
                ]
            },
        }
        result = make_frictions(input_data)
        assert len(result) == 2  # structure patch + All default
        from run_anuga import defaults
        assert result[0][1] == defaults.BUILDING_MANNINGS_N
        assert result[-1] == ['All', defaults.DEFAULT_MANNINGS_N]


# --------------------------------------------------------------------------- #
# TASK-1260: Robust CRS / UTM-zone derivation via pyproj                       #
# --------------------------------------------------------------------------- #

class TestDeriveUtmZone:
    """
    TASK-1260: get_utm_geo_reference(epsg_str) must return an anuga.Geo_reference
    with the correct zone number derived via pyproj — not the fragile [-2:] slice.

    Cases:
    - Northern hemisphere UTM: EPSG:32655 (zone 55N, Australia)
    - Southern hemisphere UTM: EPSG:32755 (zone 55S, Australia - GDA94)
    - Single-digit zone: EPSG:32601 (zone 1)
    - MGA zone: EPSG:28355 (zone 55, GDA94)
    - Non-UTM EPSG: function should still extract zone from pyproj CRS authority

    get_utm_geo_reference returns an anuga.Geo_reference, so these cases require
    the optional `anuga` engine — skip the whole class when it is absent (CI runs
    `.[dev]` without anuga; localhost has it). See CLAUDE.md run_anuga CI gotcha.
    """

    pytest.importorskip("anuga")

    def test_northern_hemisphere_zone_55(self):
        from run_anuga.run_utils import get_utm_geo_reference
        geo_ref = get_utm_geo_reference("EPSG:32655")
        assert geo_ref.zone == 55

    def test_southern_hemisphere_zone_55(self):
        from run_anuga.run_utils import get_utm_geo_reference
        geo_ref = get_utm_geo_reference("EPSG:32755")
        assert geo_ref.zone == 55

    def test_zone_1(self):
        from run_anuga.run_utils import get_utm_geo_reference
        geo_ref = get_utm_geo_reference("EPSG:32601")
        assert geo_ref.zone == 1

    def test_zone_28355_mga(self):
        """EPSG:28355 is GDA94 / MGA zone 55 — should derive zone 55."""
        from run_anuga.run_utils import get_utm_geo_reference
        geo_ref = get_utm_geo_reference("EPSG:28355")
        assert geo_ref.zone == 55

    def test_zone_56(self):
        from run_anuga.run_utils import get_utm_geo_reference
        geo_ref = get_utm_geo_reference("EPSG:28356")
        assert geo_ref.zone == 56

    def test_epsg_prefix_stripped(self):
        """Should accept both 'EPSG:32655' and '32655' forms."""
        from run_anuga.run_utils import get_utm_geo_reference
        geo_ref_with_prefix = get_utm_geo_reference("EPSG:32655")
        geo_ref_bare = get_utm_geo_reference("32655")
        assert geo_ref_with_prefix.zone == geo_ref_bare.zone == 55
