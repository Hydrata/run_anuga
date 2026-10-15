"""TASK-3638 (epic 3585 W2b): culvert package parsing, the requires_capability
guard, D15 exchange regions, build-time checks, the coarse-mesh warning,
culverts/metrics.json (the pinned hydrata <-> run_anuga contract) and the
per-culvert hydraulic results (the coupler run itself is tests/test_culvert_e2e.py).

The unit tests need only numpy + shapely (both in [dev]); the real-mesh and
end-to-end tests carry requires_anuga and run in a SUBPROCESS, because a
run_sim failure ends in MPI.COMM_WORLD.Abort, which kills pytest silently.
"""

import json
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
    "regime_at_peak", "regime_fractions", "surcharge_s", "timeseries_path", "warnings",
    # TASK-3695 (additive, schema_version 1): both-end peaks + the governing-direction headwater
    "peak_hw_inlet_m", "peak_hw_outlet_m", "hw_end", "peak_hw_governing_level_m",
    "peak_hw_governing_depth_m", "peak_hw_governing_time_s",
}
HYDRAULIC_KEYS = CONTRACT_KEYS - {
    "feature_id", "name", "triangles_inlet", "triangles_outlet", "coarseness_ratio",
    "blockage_pct", "blockage_method", "timeseries_path", "warnings",
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

    def test_null_requires_is_valid(self):
        config = ScenarioConfig.model_validate(
            {"epsg": "EPSG:28355", "boundary": "b.geojson", "duration": 60, "requires": None}
        )
        assert config.requires is None

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

    def test_a_large_triangle_beside_a_fine_cluster_is_still_found(self):
        """The nearest centroids all belong to a fine cluster outside the big
        triangle that really contains the point: the full-scan fallback finds it."""
        nodes = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
        triangles = [(0, 1, 2)]
        for k in range(80):
            x = 0.2 + 0.01 * k
            base = len(nodes)
            nodes += [(x, -0.2), (x + 0.005, -0.2), (x, -0.195)]
            triangles.append((base, base + 1, base + 2))
        mesh = MeshView(nodes, triangles)
        assert mesh.containing_triangle((0.5, 0.1)) == 0
        assert mesh.local_cell_edge((0.5, 0.1)) == pytest.approx(10.0)

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

    def test_short_culvert_builds_with_a_warning(self, caplog):
        """TASK-3669 AC1+AC3: 10 m on a 5 m mesh builds; the warning is logged and kept for metrics.json."""
        mesh = grid_mesh(321000.0, 5812000.0, 20, 20, 5.0)
        culverts = load_culverts(collection(culvert_feature(coords=((321040.0, 5812050.0), (321050.0, 5812050.0)))))
        with caplog.at_level("WARNING"):
            [result] = analyse_culverts(culverts, mesh)
        short = [w for w in result.warnings if "shorter than 3 local cell edges" in w]
        assert len(short) == 1
        assert "C7" in short[0] and "10.00 m long" in short[0] and "5.00 m each" in short[0]
        assert "Suggested max triangle area near the culvert: 5.56 m2" in short[0]  # (10/3)**2 / 2
        assert short[0] in caplog.text
        [entry] = metrics_document([result])["culverts"]
        assert short[0] in entry["warnings"]

    def test_long_enough_culvert_has_no_short_warning(self):
        mesh = grid_mesh(321000.0, 5812000.0, 20, 20, 5.0)
        [result] = analyse_culverts(load_culverts(collection(culvert_feature())), mesh)
        assert not [w for w in result.warnings if "shorter than" in w]

    def test_short_culvert_with_a_region_off_the_mesh_edge_still_fails(self):
        mesh = grid_mesh(321000.0, 5812000.0, 20, 20, 5.0)
        culverts = load_culverts(collection(culvert_feature(coords=((321090.0, 5812050.0), (321100.0, 5812050.0)))))
        with pytest.raises(CulvertBuildError, match="exchange region holds 0 triangle"):
            analyse_culverts(culverts, mesh)

    def test_overlapping_exchange_regions_of_two_culverts_are_refused(self):
        """TASK-3656 AC1: side-by-side culverts share cells; the message names both culverts and ends."""
        mesh = grid_mesh(321000.0, 5812000.0, 20, 20, 5.0)
        culverts = load_culverts(collection(
            culvert_feature(fid=1, coords=((321030.0, 5812050.0), (321070.0, 5812050.0))),
            culvert_feature(fid=2, coords=((321030.0, 5812052.0), (321070.0, 5812052.0))),
        ))
        with pytest.raises(CulvertBuildError) as raised:
            analyse_culverts(culverts, mesh)
        text = str(raised.value)
        assert "C1" in text and "C2" in text and "its Inlet" in text and "its Outlet" in text
        assert "share" in text and "triangle(s)" in text

    def test_overlap_when_one_outlet_covers_another_inlet(self):
        mesh = grid_mesh(321000.0, 5812000.0, 20, 20, 5.0)
        culverts = load_culverts(collection(
            culvert_feature(fid=1, coords=((321020.0, 5812050.0), (321060.0, 5812050.0))),
            culvert_feature(fid=2, coords=((321062.0, 5812050.0), (321090.0, 5812050.0))),
        ))
        with pytest.raises(CulvertBuildError, match=r"'C1'.*\(its Outlet\) and .*'C2'.*\(its Inlet\)"):
            analyse_culverts(culverts, mesh)

    def test_well_separated_culverts_do_not_overlap(self):
        mesh = grid_mesh(321000.0, 5812000.0, 20, 20, 5.0)
        culverts = load_culverts(collection(
            culvert_feature(fid=1, coords=((321020.0, 5812020.0), (321060.0, 5812020.0))),
            culvert_feature(fid=2, coords=((321020.0, 5812080.0), (321060.0, 5812080.0))),
        ))
        assert len(analyse_culverts(culverts, mesh)) == 2

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

    def test_exchange_region_geojson_beside_metrics_matches_it(self, tmp_path):
        """TASK-3646 AC1: one polygon per end; count + ratio agree with metrics.json."""
        mesh = grid_mesh(321000.0, 5812000.0, 20, 20, 5.0)
        input_data = {
            "culverts": load_culverts(collection(culvert_feature())),
            "output_directory": str(tmp_path),
            "scenario_config": {"epsg": "EPSG:32755"},
        }
        [result] = prepare_culverts(input_data, mesh)
        metrics = json.loads((tmp_path / "culverts" / "metrics.json").read_text())["culverts"][0]
        regions = json.loads((tmp_path / "culverts" / "exchange_regions.geojson").read_text())
        assert regions["type"] == "FeatureCollection"
        assert regions["crs"]["properties"]["name"] == "EPSG:32755"
        by_end = {f["properties"]["end"]: f for f in regions["features"]}
        assert sorted(by_end) == ["inlet", "outlet"]
        for end in ("inlet", "outlet"):
            feature = by_end[end]
            props = feature["properties"]
            assert props["feature_id"] == 7 and props["name"] == "C7"
            assert props["triangle_count"] == metrics[f"triangles_{end}"] >= constants.MIN_REGION_TRIANGLES
            assert props["coarseness_ratio"] == metrics["coarseness_ratio"]
            assert props["cell_edge_m"] == pytest.approx(result.cell_edges_m[end], abs=1e-4)
            ring = feature["geometry"]["coordinates"][0]
            assert feature["geometry"]["type"] == "Polygon" and ring[0] == ring[-1] and len(ring) == 5
            assert ring == [[round(x, 3), round(y, 3)] for x, y in result.regions[end].polygon]

    def test_exchange_region_geojson_without_epsg_has_no_crs(self, tmp_path):
        mesh = grid_mesh(321000.0, 5812000.0, 20, 20, 5.0)
        input_data = {"culverts": load_culverts(collection(culvert_feature())), "output_directory": str(tmp_path)}
        prepare_culverts(input_data, mesh)
        regions = json.loads((tmp_path / "culverts" / "exchange_regions.geojson").read_text())
        assert "crs" not in regions and len(regions["features"]) == 2

    def test_no_culverts_writes_nothing(self, tmp_path):
        assert prepare_culverts({"culverts": [], "output_directory": str(tmp_path)}, None) == []
        assert not (tmp_path / "culverts").exists()


# ── real ANUGA mesh (subprocess); the end-to-end run is tests/test_culvert_e2e.py ──

SMALL_TEST_CULVERT = culvert_feature(
    fid=101, name="Small test box", coords=((251060.0, 6271900.0), (251100.0, 6271900.0)),
    barrels=1, width_m=1.5, height_m=1.0,
)


def _run_python(code, timeout=600):
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          capture_output=True, text=True, timeout=timeout)


def _run_sim_recording_terminal_errors(package, tmp_path, **run_kwargs):
    """run_sim in a subprocess (a failure ends in MPI Abort, which would kill pytest) with a
    telemetry client that records every terminal ``error`` event to a JSON list; returns that
    file's path (asserted to exist, i.e. at least one error was posted)."""
    record = tmp_path / "terminal_errors.json"
    extra = "".join(f", {key}={value!r}" for key, value in run_kwargs.items())
    result = _run_python(f"""
        import json
        from run_anuga.run import run_sim

        class Client:
            errors = []
            def error(self, message):
                self.errors.append(message)
                open({str(record)!r}, "w").write(json.dumps(self.errors))
                return True
            def __getattr__(self, name):
                return lambda *a, **k: True
        try:
            run_sim({str(package)!r}, telemetry_client=Client(){extra})
        except BaseException:
            pass
    """, timeout=900)
    assert record.exists(), result.stdout + result.stderr
    return record


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


# ── per-culvert hydraulic results (TASK-3642, pure numpy) ─────────────


class TestCulvertHydraulics:

    CONTROLS = ("barrel", "inlet", "reverse-inlet", "n/a")

    @staticmethod
    def series():
        return {
            "time_s": [0.0, 10.0, 20.0, 30.0],
            "q_m3s": [0.0, 1.0, 2.5, -0.5],
            "hw_level_m": [100.0, 100.5, 101.4, 100.2],
            "tw_level_m": [99.0, 99.2, 99.9, 100.1],
            "control": [0, 0, 1, 2],
        }

    def test_definitions(self):
        from run_anuga.culverts.outputs import culvert_hydraulics

        [culvert] = load_culverts(collection(culvert_feature(width_m=2.0, height_m=1.0, blockage_pct=50)))
        h = culvert_hydraulics(culvert, 100.0, self.series(), self.CONTROLS)
        assert set(h) == HYDRAULIC_KEYS
        assert h["peak_hw_level_m"] == 101.4 and h["peak_hw_depth_m"] == 1.4 and h["peak_hw_time_s"] == 20.0
        assert h["hw_over_d"] == pytest.approx(1.4)  # against the UNBLOCKED rise
        assert h["peak_tw_level_m"] == 100.1
        assert h["peak_q_m3s"] == 2.5 and h["peak_q_time_s"] == 20.0
        assert h["barrel_velocity_ms"] == pytest.approx(2.5 / (2.0 * 0.5 * 1.0))  # reduced-area flow area
        assert h["regime_at_peak"] == "inlet"
        assert h["regime_fractions"] == {"barrel": 0.5, "inlet": 0.25, "reverse-inlet": 0.25}
        assert h["surcharge_s"] == 10.0  # depth 1.4 > rise 1.0 for the one 10 s sample
        assert json.dumps(h)

    def test_a_reverse_peak_keeps_its_sign(self):
        from run_anuga.culverts.outputs import culvert_hydraulics

        [culvert] = load_culverts(collection(culvert_feature()))
        series = self.series()
        series["q_m3s"] = [0.0, -4.0, 1.0, 0.0]
        h = culvert_hydraulics(culvert, 100.0, series, self.CONTROLS)
        assert h["peak_q_m3s"] == -4.0 and h["peak_q_time_s"] == 10.0

    def test_reverse_dominant_culvert_reports_the_outlet_end_headwater(self):
        """TASK-3695: peak |Q| flows Outlet -> Inlet, so the headwater is at the Outlet end.
        hw_over_d and surcharge_s follow the governing (upstream-of-peak-flow) end, measured
        over THAT end's invert; the existing per-end keys keep their meaning (peak_hw_* = the
        Inlet end, peak_tw_level_m = the Outlet end) and both-end keys are added."""
        from run_anuga.culverts.outputs import culvert_hydraulics

        [culvert] = load_culverts(collection(culvert_feature(width_m=2.0, height_m=1.0)))
        series = {
            "time_s": [0.0, 10.0, 20.0, 30.0],
            "q_m3s": [0.0, -1.0, -3.0, 0.5],
            "hw_level_m": [100.0, 100.05, 100.1, 100.05],  # Inlet end: barely wet
            "tw_level_m": [99.5, 100.6, 101.3, 100.2],  # Outlet end: surcharged
            "control": [0, 2, 2, 0],
        }
        h = culvert_hydraulics(culvert, 100.0, series, self.CONTROLS, downstream_invert_m=99.5)
        assert set(h) == HYDRAULIC_KEYS
        assert h["hw_end"] == "outlet"
        assert h["peak_hw_governing_level_m"] == 101.3
        assert h["peak_hw_governing_depth_m"] == pytest.approx(1.8)  # over the OUTLET invert 99.5
        assert h["peak_hw_governing_time_s"] == 20.0
        assert h["hw_over_d"] == pytest.approx(1.8)  # governing end, not the near-dry Inlet (0.1)
        assert h["surcharge_s"] == 20.0  # Outlet depths 1.1 and 1.8 > rise 1.0: two 10 s samples
        assert h["peak_hw_inlet_m"] == 100.1 and h["peak_hw_outlet_m"] == 101.3
        # unchanged per-end meanings, which the gmc Profile draws at each end
        assert h["peak_hw_level_m"] == 100.1 and h["peak_hw_depth_m"] == pytest.approx(0.1)
        assert h["peak_tw_level_m"] == 101.3

    def test_reverse_dominant_keys_on_a_forward_series_match_the_inlet_headwater(self):
        from run_anuga.culverts.outputs import culvert_hydraulics

        [culvert] = load_culverts(collection(culvert_feature(width_m=2.0, height_m=1.0)))
        h = culvert_hydraulics(culvert, 100.0, self.series(), self.CONTROLS, downstream_invert_m=99.0)
        assert h["hw_end"] == "inlet"
        assert h["peak_hw_governing_level_m"] == h["peak_hw_level_m"] == 101.4
        assert h["peak_hw_governing_depth_m"] == h["peak_hw_depth_m"] == 1.4
        assert h["peak_hw_governing_time_s"] == h["peak_hw_time_s"] == 20.0
        assert h["hw_over_d"] == pytest.approx(1.4) and h["surcharge_s"] == 10.0
        assert h["peak_hw_inlet_m"] == 101.4 and h["peak_hw_outlet_m"] == 100.1

    def test_empty_series_gives_no_values(self):
        from run_anuga.culverts.outputs import culvert_hydraulics

        [culvert] = load_culverts(collection(culvert_feature()))
        empty = {k: [] for k in ("time_s", "q_m3s", "hw_level_m", "tw_level_m", "control")}
        assert culvert_hydraulics(culvert, 100.0, empty, self.CONTROLS) == {}

    def test_metrics_document_carries_hydraulics_path_and_run_fields(self):
        mesh = grid_mesh(321000.0, 5812000.0, 20, 20, 5.0)
        results = analyse_culverts(load_culverts(collection(culvert_feature())), mesh)
        document = metrics_document(
            results, hydraulics=[{"peak_q_m3s": 1.5, "not_a_contract_key": 9}],
            timeseries_paths=["culverts/timeseries_7.csv"], run_fields={"swmm_routing_error_pct": 0.01},
        )
        [entry] = document["culverts"]
        assert set(entry) == CONTRACT_KEYS
        assert entry["peak_q_m3s"] == 1.5 and entry["timeseries_path"] == "culverts/timeseries_7.csv"
        assert document["swmm_routing_error_pct"] == 0.01

    def test_timeseries_csv_matches_the_structure_result_columns(self, tmp_path):
        from run_anuga.culverts.outputs import write_timeseries_csv

        relative = write_timeseries_csv(str(tmp_path), 7, self.series(), self.CONTROLS)
        assert relative == "culverts/timeseries_7.csv"
        lines = (tmp_path / relative).read_text().splitlines()
        assert lines[0] == "time_s,q_m3s,hw_level_m,tw_level_m,control"
        assert len(lines) == 5 and lines[3].endswith(",inlet")

    def test_requested_flow_is_a_trailing_diagnostic_column(self, tmp_path):
        """TASK-3700: q_m3s is the moved flow; the coupler's request follows ``control``."""
        from run_anuga.culverts.outputs import write_timeseries_csv

        series = dict(self.series(), q_requested_m3s=[0.0, 4.2, 4.2, -0.5])
        relative = write_timeseries_csv(str(tmp_path), 7, series, self.CONTROLS)
        lines = (tmp_path / relative).read_text().splitlines()
        assert lines[0] == "time_s,q_m3s,hw_level_m,tw_level_m,control,q_requested_m3s"
        assert lines[2] == "10,1,100.5,99.2,barrel,4.2"

    def test_long_series_is_thinned_keeping_first_and_last(self, tmp_path, monkeypatch):
        from run_anuga.culverts.outputs import write_timeseries_csv

        monkeypatch.setattr(constants, "TIMESERIES_MAX_ROWS", 5)
        n = 100
        series = {"time_s": list(range(n)), "q_m3s": [0.0] * n, "hw_level_m": [0.0] * n,
                  "tw_level_m": [0.0] * n, "control": [0] * n}
        relative = write_timeseries_csv(str(tmp_path), 1, series, self.CONTROLS)
        rows = (tmp_path / relative).read_text().splitlines()[1:]
        assert len(rows) == 5 and rows[0].startswith("0,") and rows[-1].startswith("99,")


@pytest.mark.requires_anuga
class TestCouplerGuards:

    def test_multi_process_runs_are_refused_plainly(self, monkeypatch):
        import types

        import anuga
        from run_anuga.culverts.operator import attach_culvert_coupler

        monkeypatch.setattr(anuga, "numprocs", 2, raising=False)
        with pytest.raises(CulvertBuildError, match="multi-process"):
            attach_culvert_coupler(types.SimpleNamespace(numproc=2), [], "/nonexistent", 60)

    def test_culvert_package_refused_on_checkpoint_resume(self, small_test_copy, tmp_path):
        """TASK-3693 (operator AC0 15-10-26 = REFUSE): the coupler and SWMM state are not in
        the ANUGA checkpoint, so a resume (batch_number > 1) of a culvert package is refused
        with a plain CulvertBuildError BEFORE the checkpoint pickle is read (no pickle exists
        here: a load attempt would retry for ~50 s, then fail with 'Unable to open checkpoint
        file') and before any evolve. run_sim ends in MPI Abort, so it runs in a subprocess and
        the error is read from the telemetry client it posts to."""
        _add_culvert_layer(small_test_copy, collection(SMALL_TEST_CULVERT))
        record = _run_sim_recording_terminal_errors(small_test_copy, tmp_path, batch_number=2, checkpoint_time=30)
        [message] = json.loads(record.read_text())
        assert message.startswith("Cannot build the culverts:")
        assert "checkpoint resume" in message and "rerun" in message
        [output_dir] = small_test_copy.glob("outputs_*")
        assert not list(output_dir.glob("*.sww"))


# ── D34 buried inverts: carve to the invert + warning (TASK-3700) ──────


class _Quantity:
    def __init__(self, centroids, vertices):
        import numpy

        self.centroid_values = numpy.array(centroids, dtype=float)
        self.vertex_values = numpy.array(vertices, dtype=float)
        self.edge_values = self.vertex_values.copy()


def fake_domain(ground, vertex_ground, depth=0.0):
    """Four 10 m2 triangles; ``depth`` of water everywhere."""
    import types

    import numpy

    elevation = _Quantity(ground, vertex_ground)
    stage = _Quantity(numpy.asarray(ground) + depth, numpy.asarray(vertex_ground) + depth)
    return types.SimpleNamespace(areas=numpy.full(len(ground), 10.0),
                                 quantities={"elevation": elevation, "stage": stage})


class TestCarveToInvert:

    GROUND = [6.0, 5.5, 4.0, 7.0]
    VERTICES = [[6.1, 5.9, 6.0], [5.4, 5.6, 5.5], [3.9, 4.1, 4.0], [7.0, 7.0, 7.0]]

    def test_lowers_only_region_cells_above_the_invert_and_never_raises(self):
        import numpy

        from run_anuga.culverts.regions import carve_to_invert

        domain = fake_domain(self.GROUND, self.VERTICES)
        carve = carve_to_invert(domain, [0, 1, 2], 5.0, "inlet")
        elevation = domain.quantities["elevation"]
        assert elevation.centroid_values.tolist() == [5.0, 5.0, 4.0, 7.0]  # cell 3 is outside the region
        assert elevation.vertex_values[3].tolist() == [7.0, 7.0, 7.0]
        assert elevation.vertex_values[2].tolist() == [3.9, 4.1, 4.0]      # below the invert: untouched
        for name in ("centroid_values", "edge_values", "vertex_values"):
            before = numpy.array(self.GROUND if name == "centroid_values" else self.VERTICES)
            assert numpy.all(getattr(elevation, name) <= before)          # never raised
        assert carve.cells_lowered == 2 and carve.carved_area_m2 == 20.0
        assert carve.max_cut_m == pytest.approx(1.0)
        assert carve.carved_volume_m3 == pytest.approx(10.0 * 1.0 + 10.0 * 0.5)
        assert carve.bed_before_m == pytest.approx(15.5 / 3) and carve.bed_after_m == pytest.approx(14.0 / 3)

    def test_no_water_is_made_dry_cells_stay_dry_and_wet_cells_keep_their_depth(self):
        import numpy

        from run_anuga.culverts.regions import carve_to_invert

        for depth in (0.0, 0.3):
            domain = fake_domain(self.GROUND, self.VERTICES, depth)
            carve_to_invert(domain, [0, 1, 2], 5.0, "inlet")
            for name in ("centroid_values", "edge_values", "vertex_values"):
                h = getattr(domain.quantities["stage"], name) - getattr(domain.quantities["elevation"], name)
                assert numpy.allclose(h, depth, atol=1e-12)

    def test_is_a_no_op_when_the_invert_is_at_or_above_the_ground(self):
        from run_anuga.culverts.regions import carve_to_invert

        domain = fake_domain(self.GROUND, self.VERTICES)
        before = {name: getattr(domain.quantities["elevation"], name).copy()
                  for name in ("centroid_values", "edge_values", "vertex_values")}
        # region mean ground = 15.5 / 3 = 5.1667 m: an invert at (or within 1 mm below) the mean, or
        # above it, is not buried, even though one cell (6.0 m, vertices to 6.1) stands above it
        for invert in (6.1, 5.5, 15.5 / 3, 15.5 / 3 - 0.0009):
            carve = carve_to_invert(domain, [0, 1, 2], invert, "outlet")
            for name, values in before.items():
                assert (getattr(domain.quantities["elevation"], name) == values).all()
                assert (getattr(domain.quantities["stage"], name) == values).all()
            assert carve.cells_lowered == 0 and carve.max_cut_m == 0.0 and carve.carved_volume_m3 == 0.0
            assert carve.bed_after_m == carve.bed_before_m
        assert carve_to_invert(domain, [0, 1, 2], 15.5 / 3 - 0.0011, "outlet").cells_lowered == 2


class TestBuriedInvertWarning:

    @staticmethod
    def carve(buried_by):
        from run_anuga.culverts.regions import CarveResult

        return CarveResult(end="inlet", invert_m=5.28, bed_before_m=5.28 + buried_by, bed_after_m=5.28,
                           region_area_m2=200.0, cells_lowered=16, carved_area_m2=200.0,
                           max_cut_m=buried_by, carved_volume_m3=200.0 * buried_by)

    def test_plain_english_warning_names_the_cut_area_and_the_better_fix(self):
        from run_anuga.culverts.checks import buried_invert_warning

        [culvert] = load_culverts(collection(culvert_feature(name="Box x3")))
        text = buried_invert_warning(culvert, self.carve(0.75))
        assert text.startswith(f"{culvert.label}: the Inlet invert (5.28 m) is 0.75 m below the average ground")
        assert "carved down to it before the run: by up to 0.75 m over 200 m2 (150 m3 removed)" in text
        assert "Burning the channel into the terrain" in text and "is better" in text

    def test_no_warning_within_five_centimetres_or_above_the_ground(self):
        from run_anuga.culverts.checks import buried_invert_warning

        [culvert] = load_culverts(collection(culvert_feature()))
        assert constants.BURIED_INVERT_WARN_M == 0.05
        for buried_by in (0.05, 0.01, 0.0, -0.4):
            assert buried_invert_warning(culvert, self.carve(buried_by)) is None
        assert buried_invert_warning(culvert, self.carve(0.051))


#: The TASK-3699 localhost defect, reduced: Box x3 with inverts 5.28 -> 4.382 m under ground
#: 6.0275 / 4.7703 m (a channel the 5 m DEM does not contain), in a closed 200 x 100 m box.
BURIED_SCRIPT = """
    import json, logging, os, sys, tempfile
    import numpy as np, anuga
    from run_anuga.culverts import MeshView, analyse_culverts, load_culverts
    from run_anuga.culverts.operator import attach_culvert_coupler
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    inflow_q, finaltime = {inflow_q!r}, {finaltime!r}
    work = tempfile.mkdtemp(prefix="buried_")
    os.chdir(work)
    d = anuga.rectangular_cross_domain(40, 20, len1=200.0, len2=100.0)
    d.set_name("buried"); d.set_datadir(work); d.set_store(False)
    bed = lambda x, y: np.where(x < 95.0, 6.0275, 4.7703)
    d.set_quantity("elevation", bed, location="centroids")
    d.set_quantity("friction", 0.04)
    d.set_quantity("stage", 0.0)
    d.quantities["stage"].maximum(d.quantities["elevation"])  # dry start, as run.py's protection
    d.set_boundary({{k: anuga.Reflective_boundary(d) for k in d.get_boundary_tags()}})
    feature = {{"type": "Feature", "id": 1, "geometry": {{"type": "LineString",
               "coordinates": [[70.0, 50.0], [120.0, 50.0]]}},
               "properties": {{"name": "Box x3", "shape": "box", "width_m": 1.5, "height_m": 1.2, "barrels": 3,
                              "upstream_invert_m": 5.28, "downstream_invert_m": 4.382}}}}
    culverts = load_culverts({{"type": "FeatureCollection", "features": [feature]}})
    results = analyse_culverts(culverts, MeshView.from_domain(d))
    e0 = d.quantities["elevation"].centroid_values.copy()
    v_before = float(d.get_water_volume())
    cs = attach_culvert_coupler(d, results, work, finaltime)
    v0 = float(d.get_water_volume())
    lowered = d.quantities["elevation"].centroid_values < e0
    inflow = None
    if inflow_q:
        inflow = anuga.Inlet_operator(d, anuga.Region(d, polygon=[[5, 40], [15, 40], [15, 60], [5, 60]]), Q=inflow_q)
    for _ in d.evolve(yieldstep=30.0, finaltime=finaltime):
        pass
    s = cs.timeseries(0)
    print(json.dumps(dict(
        v_before=v_before, v0=v0, v1=float(d.get_water_volume()),
        inflow=float(inflow.total_applied_volume) if inflow else 0.0,
        raised=bool((d.quantities["elevation"].centroid_values > e0).any()),
        lowered=int(lowered.sum()), lowered_x=d.centroid_coordinates[lowered][:, 0].tolist(),
        warnings=results[0].warnings, bed_in=float(cs.bed_in[0]), bed_out=float(cs.bed_out[0]),
        t=s["time_s"].tolist(), q=s["q_m3s"].tolist(), q_req=s["q_requested_m3s"].tolist(),
        hw=s["hw_level_m"].tolist(), tw=s["tw_level_m"].tolist(),
        applied=float(cs.applied_volume[0]), requested=float(cs.requested_volume[0]))))
    cs.finalize()
"""


def _buried(inflow_q, finaltime):
    result = _run_python(BURIED_SCRIPT.format(inflow_q=inflow_q, finaltime=finaltime))
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1]), result.stderr


@pytest.mark.requires_anuga
class TestBuriedInvertCarve:
    """AC2/AC3 on the real coupler (subprocess: an ANUGA/MPI abort kills pytest)."""

    def test_dry_carved_regions_read_zero_head_and_carry_zero_flow(self):
        r, log = _buried(0.0, 120.0)
        assert r["v_before"] == r["v0"] == r["v1"] == 0.0           # the carve made no water
        assert not r["raised"] and r["lowered"] > 0
        # only the two exchange regions moved: inlet x < 70, outlet x > 120
        assert all(x < 70.0 or x > 120.0 for x in r["lowered_x"])
        assert r["bed_in"] == pytest.approx(5.28, abs=1e-9) and r["bed_out"] == pytest.approx(4.382, abs=1e-9)
        assert max(abs(h - 5.28) for h in r["hw"]) < 1e-9             # zero headwater depth
        assert max(map(abs, r["q"])) == 0.0 and r["applied"] == 0.0   # zero moved flow ...
        assert max(map(abs, r["q_req"])) < 1e-3                       # ... and SWMM asks ~nothing
        # AC3: both ends logged; the 0.75 m / 0.39 m burials warn in plain English
        assert log.count("culvert carve: ") == 2 and "8 of 8 cell(s) lowered, max cut 0.747 m" in log
        assert len(r["warnings"]) == 2 and all("Burning the channel" in w for w in r["warnings"])

    def test_the_carved_pool_fills_before_the_culvert_flows(self):
        r, _ = _buried(0.5, 900.0)
        flowing = [i for i, q in enumerate(r["q"]) if q != 0.0]
        assert flowing, "the culvert never flowed"
        first = flowing[0]
        assert r["t"][first] > 300.0      # the inflow (60 m upstream) had to reach and fill the pool first
        # it starts as the pool stage passes the INVERT (not the old ground, 0.75 m higher), from ~zero head
        assert 5.28 < r["hw"][first] < 5.28 + 0.05 and 0.0 < r["q"][first] < 0.01
        for q, q_req, hw in zip(r["q"], r["q_req"], r["hw"]):
            if hw <= 5.28 + 1e-9:
                assert q == 0.0 and abs(q_req) < 1e-3                  # no head, no flow
        assert max(r["q"]) > 0.05 and min(r["q"]) >= -1e-6            # then it carries flow, forward
        assert r["v1"] - r["v0"] == pytest.approx(r["inflow"], rel=1e-6)
