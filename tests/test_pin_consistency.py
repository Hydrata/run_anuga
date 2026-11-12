"""The culvert engine (anuga_drainage) is pinned in TWO places that must agree.

- pyproject.toml's ``[sim]`` extra: what the web/editable installs and CI use.
- batch/Dockerfile's ``ARG ANUGA_DRAINAGE_SHA``: what the GPU/CPU Batch image bakes,
  i.e. what prod simulations actually run.

TASK-4356 (epic 3754): the coupler fix moved the pyproject pin only; the image would have
been rebaked with the old, buggy coupler while every test passed. This test fails if the
two pins ever diverge.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pyproject_drainage_sha():
    # Regex, not tomllib: CI still runs Python 3.10.
    text = (ROOT / "pyproject.toml").read_text()
    pins = re.findall(r'^\s*"anuga_drainage\s*@\s*git\+\S+?@([0-9a-f]+)"', text, flags=re.M)
    assert len(pins) == 1, f"expected one anuga_drainage git pin in pyproject.toml, found {pins}"
    assert len(pins[0]) == 40, f"pyproject anuga_drainage pin is not a full SHA: {pins[0]!r}"
    return pins[0]


def _dockerfile_drainage_sha():
    text = (ROOT / "batch" / "Dockerfile").read_text()
    shas = re.findall(r"^ARG ANUGA_DRAINAGE_SHA=([0-9a-f]+)\s*$", text, flags=re.M)
    assert len(shas) == 1, f"expected one defaulted ARG ANUGA_DRAINAGE_SHA, found {shas}"
    assert len(shas[0]) == 40, f"Dockerfile ANUGA_DRAINAGE_SHA is not a full SHA: {shas[0]!r}"
    return shas[0]


def test_batch_image_bakes_the_same_anuga_drainage_as_the_sim_extra():
    assert _dockerfile_drainage_sha() == _pyproject_drainage_sha(), (
        "batch/Dockerfile ARG ANUGA_DRAINAGE_SHA and pyproject [sim] anuga_drainage pin differ: "
        "move both together, or the Batch image runs a different culvert coupler than CI tested"
    )
