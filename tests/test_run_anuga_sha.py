"""TASK-3775: run_summary environment carries run_anuga_sha.

Source order: baked BUILD_PROVENANCE.json (7-40 hex, recorded as given; the
bake writes `rev-parse --short`) -> git HEAD of the checkout the package was
imported from (full 40-hex) -> 'unknown'.
"""

import json
import os
import re
import subprocess

import pytest

import run_anuga
from run_anuga.diagnostics import resolve_run_anuga_sha

PKG_DIR = os.path.dirname(os.path.abspath(run_anuga.__file__))
BAKED = "0123456789abcdef0123456789abcdef01234567"


def _manifest(tmp_path, sha):
    path = tmp_path / "BUILD_PROVENANCE.json"
    path.write_text(json.dumps({
        "run_anuga": {"git_url": "https://github.com/Hydrata/run_anuga", "sha": sha},
        "anuga_core": {"git_url": "x", "sha": "abc"},
        "hydrata": {"git_url": "y", "sha": "def"},
    }))
    return str(path)


def _git_head():
    out = subprocess.run(["git", "-C", PKG_DIR, "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def test_baked_manifest_wins(tmp_path):
    assert resolve_run_anuga_sha(_manifest(tmp_path, BAKED)) == (BAKED, "build_provenance")


def test_without_manifest_falls_back_to_git_head(tmp_path):
    head = _git_head()
    assert head is not None, "the tests run from a run_anuga git checkout"
    sha, source = resolve_run_anuga_sha(str(tmp_path / "absent.json"))
    assert (sha, source) == (head, "git")


@pytest.mark.parametrize("short", ["a1374f2", "a1374f289af7", "A1374F2"])
def test_a_short_manifest_sha_is_recorded_as_given(tmp_path, short):
    # The bake tooling stamps `git rev-parse --short` (prod GPU image
    # gpu-3772 carries run_anuga.sha == 'a1374f2'). It must be reported, not
    # fall through to git (no .git in the container) and come out 'unknown'.
    no_git = str(tmp_path / "pkg")
    assert resolve_run_anuga_sha(_manifest(tmp_path, short), package_dir=no_git) == (
        short.lower(), "build_provenance")


@pytest.mark.parametrize("bad", ["a1374f", "abc", "nogit", "a1374fz", "g1374f2", "",
                                 "0123456789abcdef0123456789abcdef012345678"])
def test_a_non_hex_or_too_short_manifest_sha_is_ignored(tmp_path, bad):
    # <7 chars, non-hex, the bake's 'nogit' fallback, empty, or >40: not a sha.
    no_git = str(tmp_path / "pkg")
    assert resolve_run_anuga_sha(_manifest(tmp_path, bad), package_dir=no_git) == ("unknown", "unknown")


def test_an_ignored_manifest_sha_still_falls_back_to_git(tmp_path):
    head = _git_head()
    assert head is not None, "the tests run from a run_anuga git checkout"
    assert resolve_run_anuga_sha(_manifest(tmp_path, "nogit")) == (head, "git")


def test_no_manifest_and_no_git_is_unknown(tmp_path):
    pkg = tmp_path / "site-packages" / "run_anuga"
    pkg.mkdir(parents=True)
    assert resolve_run_anuga_sha(str(tmp_path / "absent.json"), package_dir=str(pkg)) == ("unknown", "unknown")


def test_a_package_inside_some_other_repo_does_not_report_that_repos_sha(tmp_path):
    repo = tmp_path / "other"
    pkg = repo / "venv" / "site-packages" / "run_anuga"
    pkg.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    assert resolve_run_anuga_sha(str(tmp_path / "absent.json"), package_dir=str(pkg)) == ("unknown", "unknown")


@pytest.mark.parametrize("baked", [BAKED, "a1374f2"])
def test_environment_block_carries_the_sha_next_to_the_version(tmp_path, monkeypatch, baked):
    from tests.test_diagnostics import _make_mock_domain
    from run_anuga import _handoff
    from run_anuga.diagnostics import SimulationMonitor

    monkeypatch.setattr(_handoff, "BUILD_PROVENANCE_PATH", _manifest(tmp_path, baked))
    mon = SimulationMonitor(_make_mock_domain(), str(tmp_path), 1, 10.0, duration_s=10.0,
                            run_label="r", scenario_config={})
    mon.record(10.0, wall_time_s=1.0)
    mon.finalize()
    env = json.load(open(tmp_path / "run_summary_1.json"))["environment"]
    assert "run_anuga_version" in env
    assert env["run_anuga_sha"] == baked and env["run_anuga_sha_source"] == "build_provenance"


def test_environment_sha_is_40_hex_or_unknown_without_a_manifest(tmp_path, monkeypatch):
    from tests.test_diagnostics import _make_mock_domain
    from run_anuga import _handoff
    from run_anuga.diagnostics import SimulationMonitor

    monkeypatch.setattr(_handoff, "BUILD_PROVENANCE_PATH", str(tmp_path / "absent.json"))
    mon = SimulationMonitor(_make_mock_domain(), str(tmp_path), 1, 10.0, duration_s=10.0,
                            run_label="r", scenario_config={})
    mon.record(10.0, wall_time_s=1.0)
    mon.finalize()
    sha = json.load(open(tmp_path / "run_summary_1.json"))["environment"]["run_anuga_sha"]
    assert sha == "unknown" or re.fullmatch(r"[0-9a-f]{40}", sha)
