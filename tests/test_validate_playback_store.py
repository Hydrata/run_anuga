"""Tests for the validate_playback_store CLI wrapper (main()).

validate_store() itself is exercised in depth by
tests/test_playback_store.py::TestValidatorCatchesRealDefects (a valid
store + several deliberately-broken ones) — this file covers only the CLI
argument handling / exit codes.
"""
import pytest

from run_anuga.validate_playback_store import main

try:
    import zarr  # noqa: F401

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

requires_zarr = pytest.mark.skipif(not HAS_ZARR, reason="zarr not installed")


def test_no_args_returns_usage_error(capsys):
    rc = main([])
    assert rc == 2
    assert "usage" in capsys.readouterr().err


def test_too_many_args_returns_usage_error(capsys):
    rc = main(["a", "b"])
    assert rc == 2


@requires_zarr
def test_valid_store_returns_zero(tmp_path, capsys):
    import zarr
    from zarr.codecs import BytesCodec, GzipCodec

    # Minimal but fully-conformant single-array store isn't realistic (the
    # validator requires all named arrays) — build via the real exporter
    # instead for a true positive.
    from run_anuga import playback_store as ps
    import os

    fixture_sww = os.path.join(os.path.dirname(__file__), "..", "domain.sww")
    result = ps.export_playback_store(
        input_data={
            "run_label": "run_1_1_1",
            "scenario_config": {"project": 1, "id": 1, "run_id": 1, "epsg": "EPSG:28355"},
        },
        sww_path=fixture_sww,
        output_dir=str(tmp_path),
        upload=False,
    )
    rc = main([result["local_path"]])
    assert rc == 0
    assert "VALID" in capsys.readouterr().out


@requires_zarr
def test_invalid_store_returns_one(tmp_path, capsys):
    import zarr

    store_path = tmp_path / "empty.zarr"
    zarr.open_group(str(store_path), mode="w", zarr_format=3)
    rc = main([str(store_path)])
    assert rc == 1
    assert "INVALID" in capsys.readouterr().out
