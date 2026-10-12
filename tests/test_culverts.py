"""TASK-3638 (epic 3585 W2b): culvert package parsing, the requires_capability
guard, D15 exchange regions, build-time checks, the coarse-mesh warning,
culverts/metrics.json (the pinned hydrata <-> run_anuga contract) and the stub
operator.

The unit tests need only numpy + shapely (both in [dev]); the real-mesh and
end-to-end tests carry requires_anuga and run in a SUBPROCESS, because a
run_sim failure ends in MPI.COMM_WORLD.Abort, which kills pytest silently.
"""

import json
import math
import subprocess
import sys
import textwrap

import pytest

from run_anuga.config import (
    ScenarioConfig,
    UnsupportedCapabilityError,
    check_required_capabilities,
)
from run_anuga.culverts import (
    CulvertBuildError,
    MeshView,
    analyse_culverts,
    check_culvert_placement,
    load_culverts,
    prepare_culverts,
)
from run_anuga.culverts import constants
from run_anuga.culverts.checks import coarse_mesh_warning, suggested_max_triangle_area_m2
from run_anuga.culverts.metrics import metrics_document
from run_anuga.culverts.regions import build_exchange_region
from run_anuga.run_utils import _load_package_data

CONTRACT_KEYS = {
    "feature_id", "name", "triangles_inlet", "triangles_outlet", "coarseness_ratio",
    "blockage_pct", "blockage_method", "peak_hw_level_m", "peak_hw_depth_m", "peak_hw_time_s",
    "hw_over_d", "peak_tw_level_m", "peak_q_m3s", "peak_q_time_s", "barrel_velocity_ms",
    "regime_at_peak", "regime_fractions", "surcharge_s", "timeseries_path",
}
HYDRAULIC_KEYS = CONTRACT_KEYS - {
    "feature_id", "name", "triangles_inlet", "triangles_outlet", "coarseness_ratio",
    "blockage_pct", "blockage_method", "timeseries_path",
}


def culvert_feature(fid=7, coords=((321030.0, 5812050.0), (321070.0, 5812050.0)), **props):
    properties = {"name": f"C{fid}", "shape": "box", "width_m": 1.5, "height_m": 1.0}
    properties.update(props)
    return {
        "type": "Feature",
        "id": fid,
        "geometry": {"type": "LineString", "coordinates": [list(c) for c in coords]},
        "properties": properties,
    }


def collection(*features):
    return {"type": "FeatureCollection", "features": list(features)}


def grid_mesh(x0, y0, nx, ny, edge):
    """A structured right-isosceles triangle mesh: every triangle has area edge**2 / 2."""
    nodes = [(x0 + i * edge, y0 + j * edge) for j in range(ny + 1) for i in range(nx + 1)]
    triangles = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b, c, d = a + 1, a + nx + 1, a + nx + 2
            triangles += [(a, b, d), (a, d, c)]
    return MeshView(nodes, triangles)


# ── requires_capability guard ─────────────────────────────────────────


class TestRequiresCapability:

    def test_unknown_capability_is_refused_by_name(self):
        with pytest.raises(UnsupportedCapabilityError, match="'nonexistent'"):
            check_required_capabilities({"requires": ["nonexistent"]})

    def test_culvert_is_supported(self):
        assert check_required_capabilities({"requires": ["culvert"]}) == ["culvert"]

    def test_absent_or_empty_requires_passes(self):
        assert check_required_capabilities({}) == []
        assert check_required_capabilities({"requires": []}) == []

    def test_a_bare_string_is_refused(self):
        with pytest.raises(UnsupportedCapabilityError, match="must be a list"):
            check_required_capabilities({"requires": "culvert"})

    def test_scenario_config_rejects_unknown_capability(self):
        with pytest.raises(ValueError, match="'nonexistent'"):
            ScenarioConfig.model_validate(
                {"epsg": "EPSG:28355", "boundary": "b.geojson", "duration": 60, "requires": ["nonexistent"]}
            )

    def test_package_load_refuses_unknown_capability(self, scenario_package):
        _set_scenario(scenario_package, requires=["nonexistent"])
        with pytest.raises(UnsupportedCapabilityError, match="'nonexistent'"):
            _load_package_data(str(scenario_package))

    def test_package_requiring_culvert_without_the_layer_is_refused(self, scenario_package):
        _set_scenario(scenario_package, requires=["culvert"])
        with pytest.raises(CulvertBuildError, match="no culvert layer"):
            _load_package_data(str(scenario_package))

    def test_package_requiring_culvert_loads_the_culverts(self, scenario_package):
        _add_culvert_layer(scenario_package, collection(culvert_feature()))
        input_data = _load_package_data(str(scenario_package))
        assert [c.feature_id for c in input_data["culverts"]] == [7]

    def test_package_without_culverts_has_none(self, scenario_package):
        assert _load_package_data(str(scenario_package))["culverts"] == []

    @pytest.mark.parametrize("command", ["validate", "run"])
    def test_cli_exits_non_zero_naming_the_capability(self, scenario_package, command):
        _set_scenario(scenario_package, requires=["nonexistent"])
        result = subprocess.run(
            [sys.executable, "-m", "run_anuga.cli", command, str(scenario_package)],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode != 0
        assert "'nonexistent'" in result.stderr


def _set_scenario(package, **updates):
    path = package / "scenario.json"
    config = json.loads(path.read_text())
    config.update(updates)
    path.write_text(json.dumps(config))


def _add_culvert_layer(package, layer):
    (package / "inputs" / "culvert.geojson").write_text(json.dumps(layer))
    _set_scenario(package, culvert="culvert.geojson", requires=["culvert"])


# ── culvert.geojson parsing ───────────────────────────────────────────


class TestLoadCulverts:

    def test_defaults_are_filled(self):
        [culvert] = load_culverts(collection(culvert_feature(barrels=None, wall_thickness_m=None)))
        assert culvert.barrels == 1
        assert culvert.wall_thickness_m == pytest.approx(0.20)
        assert culvert.blockage_pct == 0.0
        assert culvert.name == "C7"

    def test_footprint_width_counts_every_barrel_and_wall(self):
        [culvert] = load_culverts(collection(culvert_feature(barrels=3, width_m=1.2, wall_thickness_m=0.2)))
        assert culvert.footprint_width_m == pytest.approx(3 * 1.2 + 4 * 0.2)

    def test_circular_uses_the_diameter(self):
        [culvert] = load_culverts(collection(culvert_feature(shape="circular", diameter_m=0.9)))
        assert culvert.barrel_width_m == pytest.approx(0.9)
        assert culvert.rise_m == pytest.approx(0.9)

    def test_ends_follow_the_drawn_direction(self):
        [culvert] = load_culverts(collection(culvert_feature()))
        assert culvert.inlet_xy == (321030.0, 5812050.0)
        assert culvert.outlet_xy == (321070.0, 5812050.0)
        assert culvert.outward_direction("inlet") == (-1.0, 0.0)
        assert culvert.outward_direction("outlet") == (1.0, 0.0)
        assert culvert.length_m == pytest.approx(40.0)

    def test_length_override_wins(self):
        [culvert] = load_culverts(collection(culvert_feature(length_override_m=55)))
        assert culvert.length_m == 55.0

    def test_every_bad_feature_is_named_at_once(self):
        layer = collection(
            culvert_feature(fid=1, name="Road", shape="oval"),
            culvert_feature(fid=2, name="Rail", barrels=21),
            culvert_feature(fid=3, name="Track", coords=((1.0, 1.0),)),
            culvert_feature(fid=4, name="Lane", shape="circular", diameter_m=None),
        )
        with pytest.raises(CulvertBuildError) as error:
            load_culverts(layer)
        message = str(error.value)
        for expected in ("'Road'", "box or circular", "'Rail'", "between 1 and 20", "'Track'",
                         "two distinct points", "'Lane'", "diameter_m"):
            assert expected in message

    def test_missing_layer_is_empty(self):
        assert load_culverts(None) == []


# ── placement (pre-mesh) ──────────────────────────────────────────────

OUTLINE = [[321000.0, 5812000.0], [321100.0, 5812000.0], [321100.0, 5812100.0], [321000.0, 5812100.0]]


def building(method, ring, fid="b1"):
    return {"type": "Feature", "id": fid, "properties": {"method": method},
            "geometry": {"type": "Polygon", "coordinates": [ring]}}


SQUARE_AT_OUTLET = [[321065, 5812045], [321075, 5812045], [321075, 5812055], [321065, 5812055], [321065, 5812045]]


class TestPlacement:

    def test_ends_inside_the_outline_pass(self):
        check_culvert_placement(load_culverts(collection(culvert_feature())), OUTLINE)

    def test_end_outside_the_outline_fails_plainly(self):
        culverts = load_culverts(collection(culvert_feature(coords=((321030, 5812050), (321130, 5812050)))))
        with pytest.raises(CulvertBuildError, match="Outlet at .* is outside the model outline"):
            check_culvert_placement(culverts, OUTLINE)

    def test_end_inside_a_reflective_building_fails_plainly(self):
        culverts = load_culverts(collection(culvert_feature()))
        buildings = collection(building("Reflective", SQUARE_AT_OUTLET))
        with pytest.raises(CulvertBuildError, match="Outlet at .* is inside Reflective building b1"):
            check_culvert_placement(culverts, OUTLINE, buildings)

    def test_end_inside_a_mannings_building_passes(self):
        culverts = load_culverts(collection(culvert_feature()))
        check_culvert_placement(culverts, OUTLINE, collection(building("Mannings", SQUARE_AT_OUTLET)))


# ── mesh analysis (D15) on a synthetic mesh ───────────────────────────


class TestExchangeRegions:

    def test_local_cell_edge_matches_the_resolution_convention(self):
        mesh = grid_mesh(321000.0, 5812000.0, 20, 20, 5.0)
        assert mesh.local_cell_edge((321032.0, 5812051.0)) == pytest.approx(5.0)
        assert mesh.local_cell_edge((320000.0, 5812051.0)) is None

    def test_regions_sit_on_the_approach_side_with_at_least_four_triangles(self):
        mesh = grid_mesh(321000.0, 5812000.0, 20, 20, 5.0)
        [result] = analyse_culverts(load_culverts(collection(culvert_feature())), mesh)
        for end, region in result.regions.items():
            assert region.triangle_count >= constants.MIN_REGION_TRIANGLES
            xs = [mesh.centroids[i][0] for i in region.triangle_ids]
            if end == "inlet":
                assert max(xs) <= 321030.0  # upstream of the inlet, outside the barrel
            else:
                assert min(xs) >= 321070.0
        # width = max(footprint 1.9 m, 2 cell edges) and depth = 2 cell edges
        assert result.inlet_region.width_m == pytest.approx(10.0)
        assert result.inlet_region.depth_m == pytest.approx(10.0)

    def test_wide_culvert_region_takes_the_footprint_width(self):
        mesh = grid_mesh(321000.0, 5812000.0, 20, 20, 5.0)
        [result] = analyse_culverts(load_culverts(collection(culvert_feature(barrels=8, width_m=2.0))), mesh)
        assert result.inlet_region.width_m == pytest.approx(8 * 2.0 + 9 * 0.2)

    def test_region_expands_until_it_holds_four_triangles(self):
        mesh = grid_mesh(321000.0, 5812000.0, 20, 20, 5.0)
        region = build_exchange_region(mesh, "inlet", (321030.0, 5812050.0), (-1.0, 0.0), 0.5, 1.0)
        assert region.expansions > 0
        assert region.triangle_count >= constants.MIN_REGION_TRIANGLES

    def test_end_on_the_model_edge_facing_out_fails_the_build(self):
        mesh = grid_mesh(321000.0, 5812000.0, 20, 20, 5.0)
        culverts = load_culverts(collection(culvert_feature(coords=((321050.0, 5812050.0), (321100.0, 5812050.0)))))
        with pytest.raises(CulvertBuildError, match="Outlet exchange region holds 0 triangle"):
            analyse_culverts(culverts, mesh)

    def test_too_short_for_the_mesh_fails_the_build(self):
        mesh = grid_mesh(321000.0, 5812000.0, 20, 20, 5.0)
        culverts = load_culverts(collection(culvert_feature(coords=((321040.0, 5812050.0), (321052.0, 5812050.0)))))
        with pytest.raises(CulvertBuildError, match="shorter than 3 local cell edges"):
            analyse_culverts(culverts, mesh)

    def test_end_off_the_mesh_fails_the_build(self):
        mesh = grid_mesh(321000.0, 5812000.0, 10, 20, 5.0)  # x only to 321050
        culverts = load_culverts(collection(culvert_feature()))
        with pytest.raises(CulvertBuildError, match="Outlet at .* is not on the mesh"):
            analyse_culverts(culverts, mesh)


# ── coarse-mesh warning ───────────────────────────────────────────────


class TestCoarseMeshWarning:

    def test_warning_names_the_culvert_ratio_and_suggested_area(self, caplog):
        mesh = grid_mesh(321000.0, 5812000.0, 20, 20, 5.0)
        [culvert] = load_culverts(collection(culvert_feature(name="Smith St")))
        with caplog.at_level("WARNING"):
            [result] = analyse_culverts([culvert], mesh)
        ratio = 5.0 / culvert.footprint_width_m
        assert result.coarseness_ratio == pytest.approx(ratio)
        assert len(result.warnings) == 2  # both ends
        text = result.warnings[0]
        assert "'Smith St'" in text
        assert f"{ratio:.2f}" in text
        assert f"{suggested_max_triangle_area_m2(culvert):.2f} m2" in text
        assert "is coarse" in text  # 5 / 1.9 = 2.63: above the warning line, below "unreliable"
        assert text in caplog.text

    def test_wording_tracks_the_two_thresholds(self):
        [culvert] = load_culverts(collection(culvert_feature(width_m=2.0, wall_thickness_m=0.0)))
        assert coarse_mesh_warning(culvert, "inlet", 1.9) is None
        moderate = coarse_mesh_warning(culvert, "inlet", 4.0)
        assert "is coarse at" in moderate and "results are unreliable" not in moderate
        assert "very coarse" in coarse_mesh_warning(culvert, "inlet", 7.0)

    def test_suggested_area_is_the_warning_line_edge(self):
        [culvert] = load_culverts(collection(culvert_feature(width_m=1.5, wall_thickness_m=0.2)))
        assert suggested_max_triangle_area_m2(culvert) == pytest.approx(1.9 ** 2 / 2)

    def test_fine_mesh_gives_no_warning(self):
        mesh = grid_mesh(321000.0, 5812000.0, 100, 100, 1.0)
        [result] = analyse_culverts(load_culverts(collection(culvert_feature())), mesh)
        assert result.warnings == []
        assert result.coarseness_ratio < constants.COARSE_RATIO_WARN


# ── culverts/metrics.json contract ────────────────────────────────────


class TestMetricsContract:

    def test_document_matches_the_pinned_contract(self):
        mesh = grid_mesh(321000.0, 5812000.0, 20, 20, 5.0)
        results = analyse_culverts(load_culverts(collection(culvert_feature(blockage_pct=25))), mesh)
        document = metrics_document(results)
        assert document["schema_version"] == 1
        [entry] = document["culverts"]
        assert set(entry) == CONTRACT_KEYS
        assert entry["feature_id"] == 7 and entry["name"] == "C7"
        assert entry["triangles_inlet"] == results[0].inlet_region.triangle_count >= 4
        assert entry["coarseness_ratio"] == pytest.approx(5.0 / 1.9, abs=1e-4)
        assert entry["blockage_pct"] == 25.0 and entry["blockage_method"] == "reduced-area"
        assert all(entry[key] is None for key in HYDRAULIC_KEYS)
        assert entry["timeseries_path"] is None

    def test_prepare_writes_the_file_next_to_the_sww(self, tmp_path):
        mesh = grid_mesh(321000.0, 5812000.0, 20, 20, 5.0)
        input_data = {"culverts": load_culverts(collection(culvert_feature())), "output_directory": str(tmp_path)}
        prepare_culverts(input_data, mesh)
        written = json.loads((tmp_path / "culverts" / "metrics.json").read_text())
        assert written["culverts"][0]["feature_id"] == 7
        assert json.dumps(written)  # plain JSON types only

    def test_no_culverts_writes_nothing(self, tmp_path):
        assert prepare_culverts({"culverts": [], "output_directory": str(tmp_path)}, None) == []
        assert not (tmp_path / "culverts").exists()


# ── real ANUGA mesh + end-to-end stub run (subprocess) ────────────────

SMALL_TEST_CULVERT = culvert_feature(
    fid=101, name="Small test box", coords=((251060.0, 6271900.0), (251100.0, 6271900.0)),
    barrels=1, width_m=1.5, height_m=1.0,
)


def _run_python(code, timeout=600):
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          capture_output=True, text=True, timeout=timeout)


@pytest.mark.requires_anuga
class TestRealMesh:

    def test_five_metre_mesh_regions_hold_four_triangles(self, small_test_copy, tmp_path):
        """AC2: a 1.5 m culvert on a real 5 m ANUGA mesh (pmesh AND Domain views agree)."""
        _add_culvert_layer(small_test_copy, collection(SMALL_TEST_CULVERT))
        result = _run_python(f"""
            import json, anuga
            from run_anuga.run_utils import setup_input_data, create_anuga_mesh
            from run_anuga.culverts import MeshView, analyse_culverts
            data = setup_input_data({str(small_test_copy)!r})
            assert data['scenario_config']['resolution'] == 5
            _, pmesh = create_anuga_mesh(data)
            a = analyse_culverts(data['culverts'], MeshView.from_pmesh(pmesh))[0]
            domain = anuga.Domain(data['mesh_filepath'])
            b = analyse_culverts(data['culverts'], MeshView.from_domain(domain))[0]
            print(json.dumps({{
                'pmesh': [a.inlet_region.triangle_count, a.outlet_region.triangle_count],
                'domain': [b.inlet_region.triangle_count, b.outlet_region.triangle_count],
                'edges': list(b.cell_edges_m.values()), 'ratio': b.coarseness_ratio,
                'warnings': b.warnings}}))
        """)
        assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
        report = json.loads(result.stdout.strip().splitlines()[-1])
        assert min(report["domain"]) >= 4
        assert report["pmesh"] == report["domain"]
        assert all(3.0 < edge < 8.0 for edge in report["edges"])
        assert report["ratio"] > constants.COARSE_RATIO_WARN  # ~5 m cells vs a 1.9 m footprint
        assert "Small test box" in report["warnings"][0]


@pytest.mark.requires_anuga
class TestStubRunEndToEnd:

    def test_run_with_a_culvert_writes_metrics_and_logs_the_stub(self, small_test_copy):
        _add_culvert_layer(small_test_copy, collection(SMALL_TEST_CULVERT))
        _set_scenario(small_test_copy, duration=60)
        result = subprocess.run(
            [sys.executable, "-m", "run_anuga.cli", "run", str(small_test_copy)],
            capture_output=True, text=True, timeout=600,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output[-3000:]
        [output_dir] = small_test_copy.glob("outputs_*")
        assert list(output_dir.glob("*.sww"))
        document = json.loads((output_dir / "culverts" / "metrics.json").read_text())
        [entry] = document["culverts"]
        assert set(entry) == CONTRACT_KEYS
        assert entry["feature_id"] == 101 and entry["name"] == "Small test box"
        assert entry["triangles_inlet"] >= 4 and entry["triangles_outlet"] >= 4
        assert math.isfinite(entry["coarseness_ratio"])
        assert all(entry[key] is None for key in HYDRAULIC_KEYS)
        run_log = "\n".join(p.read_text() for p in output_dir.glob("run_anuga_*.log")) + output
        assert "culvert stub operator: Culvert 'Small test box' (feature 101)" in run_log
        assert "coarseness ratio" in run_log

    def test_end_inside_a_reflective_building_stops_the_run(self, small_test_copy):
        """AC3 through the real runner: plain message, non-zero exit, before the mesh."""
        _add_culvert_layer(small_test_copy, collection(SMALL_TEST_CULVERT))
        (small_test_copy / "inputs" / "building.geojson").write_text(json.dumps(collection(building(
            "Reflective", [[251095, 6271895], [251110, 6271895], [251110, 6271910], [251095, 6271910],
                           [251095, 6271895]], fid="shed"))))
        _set_scenario(small_test_copy, building="building.geojson", structure=None, duration=60)
        result = subprocess.run(
            [sys.executable, "-m", "run_anuga.cli", "run", str(small_test_copy)],
            capture_output=True, text=True, timeout=600,
        )
        assert result.returncode != 0
        assert "is inside Reflective building shed" in result.stdout + result.stderr
        assert not list(small_test_copy.glob("outputs_*/*.msh"))
