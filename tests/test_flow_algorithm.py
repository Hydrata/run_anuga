"""TASK-4351: the scenario's ANUGA flow algorithm (solver)."""
import json

import pytest
from run_anuga.config import UnsupportedCapabilityError, check_required_capabilities
from run_anuga.flow_algorithm import FLOW_ALGORITHMS, resolve_flow_algorithm


def _input(**scenario_config):
    return {"scenario_config": scenario_config}


class TestResolve:
    def test_absent_is_de0(self):
        assert resolve_flow_algorithm(_input()) == "DE0"
        assert resolve_flow_algorithm({}) == "DE0"
        assert resolve_flow_algorithm(_input(flow_algorithm=None)) == "DE0"

    @pytest.mark.parametrize("value", FLOW_ALGORITHMS)
    def test_a_known_value_is_returned(self, value):
        assert resolve_flow_algorithm(_input(flow_algorithm=value)) == value

    @pytest.mark.parametrize("value", ["de1", "DE3", "DE1.0", "", 1])
    def test_unknown_value_raises_naming_the_allowed_values(self, value):
        with pytest.raises(ValueError) as excinfo:
            resolve_flow_algorithm(_input(flow_algorithm=value))
        for name in FLOW_ALGORITHMS:
            assert name in str(excinfo.value)

    def test_no_env_override(self, monkeypatch):
        monkeypatch.setenv("RUN_ANUGA_FLOW_ALGORITHM", "DE1")
        assert resolve_flow_algorithm(_input()) == "DE0"


class TestRequiresGuard:
    def test_flow_algorithm_capability_is_supported(self):
        assert check_required_capabilities({"requires": ["flow_algorithm"]}) == ["flow_algorithm"]
        assert check_required_capabilities({"requires": ["culvert", "flow_algorithm"]}) == [
            "culvert", "flow_algorithm"]

    def test_unknown_capability_is_still_refused(self):
        with pytest.raises(UnsupportedCapabilityError):
            check_required_capabilities({"requires": ["flow_algorithm", "teleport"]})


def _run_small_test(package, **scenario):
    import subprocess
    import sys
    path = package / "scenario.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "duration": 60, **scenario}))
    result = subprocess.run([sys.executable, "-m", "run_anuga.cli", "run", str(package)],
                            capture_output=True, text=True, timeout=900)
    return result, result.stdout + result.stderr


@pytest.mark.requires_anuga
@pytest.mark.parametrize("value", [None, "DE1", "DE_ader2"])
def test_a_run_uses_and_records_its_flow_algorithm(small_test_copy, value):
    """The log states the solver; run_summary_1.json records it with its CFL. The DE1 case
    carries a culvert: a culvert run can pick DE1 and still moves water."""
    extra = {} if value is None else {"flow_algorithm": value}
    if value == "DE1":
        from tests.test_culverts import (
            SMALL_TEST_CULVERT,
            _add_culvert_layer,
            collection,
        )
        _add_culvert_layer(small_test_copy, collection(SMALL_TEST_CULVERT))
        extra["duration"] = 120  # the culvert e2e's duration: long enough for the coupler to move water
    result, output = _run_small_test(small_test_copy, **extra)
    assert result.returncode == 0, output[-3000:]
    [out] = small_test_copy.glob("outputs_*")
    expected = value or "DE0"
    assert f"flow algorithm: {expected}" in (out / "run_anuga_1.log").read_text()
    model = json.loads((out / "run_summary_1.json").read_text())["model"]
    assert model["flow_algorithm"] == expected
    assert model["cfl"] == (0.9 if expected == "DE0" else 1.0)
    if value == "DE1":
        metrics = json.loads((out / "culverts" / "metrics.json").read_text())
        [entry] = metrics["culverts"]
        assert abs(entry["peak_q_m3s"]) > 0, entry


@pytest.mark.requires_anuga
def test_an_unknown_flow_algorithm_stops_the_run_before_the_mesh(small_test_copy):
    result, output = _run_small_test(small_test_copy, flow_algorithm="DE9")
    assert result.returncode != 0
    assert "DE_ader2" in output
    assert not list(small_test_copy.glob("outputs_*/*.msh"))
