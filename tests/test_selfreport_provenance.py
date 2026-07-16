"""TASK-2287 (epic 2280 W3.2) — the container self-reports code provenance
from the baked manifest, corroborated by anuga.__version__.

Django-free / anuga-free by construction: the workstation's fork is unbuilt so
``import anuga`` FAILS here — every test MOCKS the manifest file read and the
installed-version lookup, never relying on a live anuga import or a real
``/opt/hydrata/BUILD_PROVENANCE.json``.

Pytest -k selectors: 'provenance', 'selfreport', 'version', 'handoff',
'diagnostics'.
"""
from __future__ import annotations

import json
import logging
import sys
import types
from unittest import mock


# ---------------------------------------------------------------------------
# diagnostics: anuga_core sha parse + installed-version corroboration
# ---------------------------------------------------------------------------

class TestAnugaCoreShaFromVersion:
    def test_parses_g_suffix(self):
        from run_anuga.diagnostics import anuga_core_sha_from_version
        assert anuga_core_sha_from_version("3.3.1.dev123+g57a64ab") == "57a64ab"

    def test_parses_full_sha_g_suffix(self):
        from run_anuga.diagnostics import anuga_core_sha_from_version
        assert anuga_core_sha_from_version(
            "3.3.0.dev1+g8abd3aefcc6df808372043a7ce1fbe523f5abbf3"
        ) == "8abd3aefcc6df808372043a7ce1fbe523f5abbf3"

    def test_none_on_plain_release(self):
        from run_anuga.diagnostics import anuga_core_sha_from_version
        assert anuga_core_sha_from_version("3.3.1") is None

    def test_none_on_unknown_or_nonstring(self):
        from run_anuga.diagnostics import anuga_core_sha_from_version
        assert anuga_core_sha_from_version("unknown") is None
        assert anuga_core_sha_from_version(None) is None
        assert anuga_core_sha_from_version(123) is None

    def test_installed_sha_reads_metadata_not_import(self):
        from run_anuga import diagnostics
        with mock.patch.object(
            diagnostics.importlib.metadata, "version",
            return_value="3.3.1.dev9+gdeadbeef",
        ):
            assert diagnostics.installed_anuga_core_sha() == "deadbeef"

    def test_installed_sha_none_on_lookup_error(self):
        from run_anuga import diagnostics
        with mock.patch.object(
            diagnostics.importlib.metadata, "version",
            side_effect=Exception("no anuga metadata"),
        ):
            assert diagnostics.installed_anuga_core_sha() is None


# ---------------------------------------------------------------------------
# _handoff.load_build_provenance: read manifest + stamp + corroborate
# ---------------------------------------------------------------------------

_FORK_MANIFEST = {
    "run_anuga": {"git_url": "https://github.com/Hydrata/run_anuga", "sha": "ra123"},
    "anuga_core": {"git_url": "https://github.com/Hydrata/anuga_core",
                   "sha": "8abd3aefcc6df808372043a7ce1fbe523f5abbf3"},
    "hydrata": {"git_url": "https://github.com/Hydrata/hydrata", "sha": "hy789"},
}


def _write_manifest(tmp_path, payload):
    p = tmp_path / "BUILD_PROVENANCE.json"
    p.write_text(json.dumps(payload))
    return str(p)


class TestLoadBuildProvenance:
    def test_reads_manifest_and_stamps_container_source(self, tmp_path):
        from run_anuga import diagnostics
        from run_anuga import _handoff
        path = _write_manifest(tmp_path, _FORK_MANIFEST)
        # corroboration is best-effort; pin it so this test asserts only the
        # read + stamp (a match, no warning).
        with mock.patch.object(diagnostics, "installed_anuga_core_sha",
                               return_value="8abd3ae"):
            prov = _handoff.load_build_provenance(path=path)
        assert prov is not None
        assert prov["provenance_source"] == "container"
        assert prov["run_anuga"]["sha"] == "ra123"
        assert prov["anuga_core"]["git_url"] == "https://github.com/Hydrata/anuga_core"

    def test_emits_canonical_schema_shape(self, tmp_path):
        # per-component {git_url, sha, source='container'} + top-level complete,
        # identical to build_code_provenance so the projected Run copy needs no
        # special-casing.
        from run_anuga import diagnostics, _handoff
        path = _write_manifest(tmp_path, _FORK_MANIFEST)
        with mock.patch.object(diagnostics, "installed_anuga_core_sha", return_value="8abd3ae"):
            prov = _handoff.load_build_provenance(path=path)
        for name in ("run_anuga", "anuga_core", "hydrata"):
            assert prov[name]["source"] == "container"
            assert prov[name]["sha"]
        assert prov["complete"] is True

    def test_incomplete_when_a_component_sha_missing(self, tmp_path):
        from run_anuga import diagnostics, _handoff
        partial = {"run_anuga": {"git_url": "u", "sha": "ra"},
                   "anuga_core": {"git_url": "u"},  # no sha
                   "hydrata": {"git_url": "u", "sha": "hy"}}
        path = _write_manifest(tmp_path, partial)
        with mock.patch.object(diagnostics, "installed_anuga_core_sha", return_value=None):
            prov = _handoff.load_build_provenance(path=path)
        assert prov["provenance_source"] == "container"
        assert prov["anuga_core"]["sha"] is None
        assert prov["complete"] is False

    def test_malformed_nonstring_sha_never_crashes(self, tmp_path):
        # A malformed manifest with a truthy NON-STRING sha (int/list) must NOT
        # crash the corroboration .lower() — the "never raises / never block a
        # sim" contract (epic 2280 §6; W3 adversarial-review P2). The sha is
        # coerced to str and the result is still a usable code_provenance dict.
        from run_anuga import diagnostics, _handoff
        bad = {"run_anuga": {"git_url": "u", "sha": ["not", "a", "string"]},
               "anuga_core": {"git_url": "u", "sha": 123},
               "hydrata": {"git_url": "u", "sha": "hy"}}
        path = _write_manifest(tmp_path, bad)
        with mock.patch.object(diagnostics, "installed_anuga_core_sha", return_value="12"):
            prov = _handoff.load_build_provenance(path=path)  # must not raise
        assert prov["provenance_source"] == "container"
        assert prov["anuga_core"]["sha"] == "123"  # coerced int -> str, no crash
        assert prov["complete"] is True  # all three shas truthy (coerced)

    def test_none_when_manifest_absent(self):
        from run_anuga import _handoff
        assert _handoff.load_build_provenance(path="/nonexistent/x.json") is None

    def test_none_when_manifest_empty_or_malformed(self, tmp_path):
        from run_anuga import _handoff
        empty = _write_manifest(tmp_path, {})
        assert _handoff.load_build_provenance(path=empty) is None
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        assert _handoff.load_build_provenance(path=str(bad)) is None

    def test_corroboration_match_no_warning(self, tmp_path, caplog):
        from run_anuga import diagnostics
        from run_anuga import _handoff
        path = _write_manifest(tmp_path, _FORK_MANIFEST)
        # installed short-sha is a PREFIX of the baked full sha -> corroborated.
        with mock.patch.object(diagnostics, "installed_anuga_core_sha",
                               return_value="8abd3ae"):
            with caplog.at_level(logging.WARNING):
                prov = _handoff.load_build_provenance(path=path)
        assert prov["anuga_core"]["sha"] == _FORK_MANIFEST["anuga_core"]["sha"]
        assert not any("does not match" in r.getMessage() for r in caplog.records)

    def test_corroboration_mismatch_warns_and_keeps_manifest(self, tmp_path, caplog):
        from run_anuga import diagnostics
        from run_anuga import _handoff
        path = _write_manifest(tmp_path, _FORK_MANIFEST)
        with mock.patch.object(diagnostics, "installed_anuga_core_sha",
                               return_value="ffffffffdeadbeef"):
            with caplog.at_level(logging.WARNING):
                prov = _handoff.load_build_provenance(path=path)
        # a mismatch WARNS but KEEPS the manifest value (never crashes/overwrites)
        assert prov["anuga_core"]["sha"] == _FORK_MANIFEST["anuga_core"]["sha"]
        assert prov["provenance_source"] == "container"
        assert any("does not match" in r.getMessage() for r in caplog.records)

    def test_never_raises_when_corroboration_lookup_explodes(self, tmp_path):
        from run_anuga import diagnostics
        from run_anuga import _handoff
        path = _write_manifest(tmp_path, _FORK_MANIFEST)
        with mock.patch.object(diagnostics, "installed_anuga_core_sha",
                               side_effect=RuntimeError("boom")):
            prov = _handoff.load_build_provenance(path=path)
        # a corroboration failure must not lose the manifest self-report
        assert prov["provenance_source"] == "container"
        assert prov["anuga_core"]["sha"] == _FORK_MANIFEST["anuga_core"]["sha"]


# ---------------------------------------------------------------------------
# _make_resource_sampler glue: manifest -> sampler code_provenance kwarg
# ---------------------------------------------------------------------------

class TestMakeResourceSamplerGlue:
    def test_passes_code_provenance_to_the_sampler(self, monkeypatch):
        """_make_resource_sampler reads the manifest via load_build_provenance
        and hands it to the ResourceSampler as code_provenance. gn_anuga is not
        on this box's path, so inject a fake sampler leaf that records kwargs."""
        from run_anuga import _handoff

        captured = {}

        class _FakeSampler:
            def __init__(self, scratch_dir, **kwargs):
                captured.update(kwargs)

        fake_pkg = types.ModuleType("gn_anuga")
        fake_bc = types.ModuleType("gn_anuga.batch_common")
        fake_rs = types.ModuleType("gn_anuga.batch_common.resource_sampler")
        fake_rs.ResourceSampler = _FakeSampler
        monkeypatch.setitem(sys.modules, "gn_anuga", fake_pkg)
        monkeypatch.setitem(sys.modules, "gn_anuga.batch_common", fake_bc)
        monkeypatch.setitem(sys.modules, "gn_anuga.batch_common.resource_sampler", fake_rs)

        sentinel = {"provenance_source": "container",
                    "run_anuga": {"git_url": "u", "sha": "ra"}}
        monkeypatch.setattr(_handoff, "load_build_provenance", lambda: sentinel)

        sampler = _handoff._make_resource_sampler(
            "/tmp/scratch", control_server="http://cs", ids={"run_id": 1},
        )
        assert sampler is not None
        assert captured.get("code_provenance") == sentinel
