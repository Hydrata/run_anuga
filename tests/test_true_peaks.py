"""TASK-3772 (epic 3585) — true per-step peaks.

The engine's Collect_max_quantities_operator (anuga_core fork) tracks the
running maxima every timestep and writes max_stage_c / max_depth_c /
max_speed_c / max_uh_c into the SWW. run_anuga attaches it on every rank
(run_anuga.peaks.attach_peak_operator) and builds the *_max.tif rasters and
the playback envelopes from those arrays.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest
from run_anuga import defaults, peaks

# ---------------------------------------------------------------------------
# attach_peak_operator — env toggle, update frequency, D39 velocity height
# ---------------------------------------------------------------------------


class _FakeDomain:
    def __init__(self, with_attr=True):
        self.calls = []
        if with_attr:
            self.max_quantities_operator = None

    def set_collect_max_quantities(self, **kwargs):
        self.calls.append(kwargs)
        if self.max_quantities_operator is None:
            self.max_quantities_operator = SimpleNamespace(**kwargs)
        return self.max_quantities_operator


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, True),
        ("", True),
        ("1", True),
        ("on", True),
        ("0", False),
        ("off", False),
        ("false", False),
        ("No", False),
    ],
)
def test_true_peaks_env_toggle(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(peaks.TRUE_PEAKS_ENV, raising=False)
    else:
        monkeypatch.setenv(peaks.TRUE_PEAKS_ENV, raw)
    assert peaks.true_peaks_enabled() is expected


@pytest.mark.parametrize("mode,freq", [(2, 1), (1, 10)])
def test_attach_uses_gpu_every_step_cpu_every_tenth(monkeypatch, mode, freq):
    monkeypatch.delenv(peaks.TRUE_PEAKS_ENV, raising=False)
    domain = _FakeDomain()
    op = peaks.attach_peak_operator(domain, mode)
    assert op is domain.max_quantities_operator
    assert domain.calls == [
        dict(
            update_frequency=freq,
            velocity_zero_height=defaults.MINIMUM_STORABLE_HEIGHT_M,
            store_to_sww=True,
        )
    ]
    assert defaults.MINIMUM_STORABLE_HEIGHT_M == 0.005  # D39


def test_attach_is_off_when_env_is_zero(monkeypatch):
    monkeypatch.setenv(peaks.TRUE_PEAKS_ENV, "0")
    domain = _FakeDomain()
    assert peaks.attach_peak_operator(domain, 2) is None
    assert domain.calls == []


def test_attach_reuses_an_operator_restored_from_a_checkpoint(monkeypatch):
    monkeypatch.delenv(peaks.TRUE_PEAKS_ENV, raising=False)
    domain = _FakeDomain()
    restored = object()
    domain.max_quantities_operator = restored
    assert peaks.attach_peak_operator(domain, 1) is restored


def test_attach_handles_a_pickle_without_the_attribute(monkeypatch):
    monkeypatch.delenv(peaks.TRUE_PEAKS_ENV, raising=False)
    domain = _FakeDomain(with_attr=False)
    assert peaks.attach_peak_operator(domain, 1) is not None


def test_attach_is_a_noop_on_an_engine_without_the_operator(monkeypatch):
    monkeypatch.delenv(peaks.TRUE_PEAKS_ENV, raising=False)
    assert peaks.attach_peak_operator(SimpleNamespace(), 1) is None


def test_run_sim_attaches_between_set_boundary_and_set_multiprocessor_mode():
    """Wiring (house style of test_gpu_mode_assertion): every rank, fresh or
    resumed — the call sits outside the batch_number==1 / myid==0 blocks."""
    import run_anuga.run as run_module

    src = open(run_module.__file__).read()
    call = "attach_peak_operator(domain, _multiprocessor_mode, log=logger)"
    assert src.count(call) == 1
    i_boundary = src.index("domain.set_boundary(")
    i_attach = src.index(call)
    i_mode = src.index("domain.set_multiprocessor_mode(_multiprocessor_mode)")
    i_evolve = src.index("domain.evolve(")
    assert i_boundary < i_attach < i_mode < i_evolve
    line = next(ln for ln in src.splitlines() if call in ln)
    # Same indentation as set_boundary: not nested in a rank-0 / batch-1 block.
    boundary_line = next(ln for ln in src.splitlines() if "domain.set_boundary(" in ln)
    indent = re.match(r"\s*", line).group(0)
    assert indent == re.match(r"\s*", boundary_line).group(0)


# ---------------------------------------------------------------------------
# centroid_to_node
# ---------------------------------------------------------------------------


def test_centroid_to_node_averages_adjacent_triangles():
    # Two triangles sharing the edge 1-2: nodes 1, 2 average both.
    volumes = np.array([[0, 1, 2], [1, 3, 2]])
    out = peaks.centroid_to_node(np.array([2.0, 4.0]), volumes, n_node=5)
    np.testing.assert_allclose(out, [2.0, 3.0, 3.0, 4.0, 0.0])


def test_centroid_to_node_skips_nan_triangles():
    volumes = np.array([[0, 1, 2], [1, 3, 2]])
    out = peaks.centroid_to_node(np.array([np.nan, 4.0]), volumes, n_node=4)
    np.testing.assert_allclose(out, [0.0, 4.0, 4.0, 4.0])


# ---------------------------------------------------------------------------
# read_engine_peaks against real SWWs
# ---------------------------------------------------------------------------


@pytest.mark.requires_anuga
def test_fixture_sww_carries_the_engine_arrays(fixture_sww):
    engine = peaks.read_engine_peaks(fixture_sww)
    assert engine is not None
    assert set(engine) == set(peaks.ENGINE_PEAK_VARIABLES)
    sizes = {v.shape for v in engine.values()}
    assert len(sizes) == 1
    assert np.all(engine["max_depth_c"] >= 0)
    assert float(engine["max_depth_c"].max()) > 0


@pytest.mark.requires_anuga
def test_old_sww_has_no_engine_arrays(fixture_sww_frames_only):
    assert peaks.read_engine_peaks(fixture_sww_frames_only) is None


@pytest.mark.requires_anuga
def test_centroid_xy_matches_plot_utils_centroids(fixture_sww):
    from anuga.utilities import plot_utils

    cx, cy = peaks.read_centroid_xy(fixture_sww)
    p = plot_utils.get_centroids(str(fixture_sww), timeSlices="last")
    np.testing.assert_allclose(cx, p.x + p.xllcorner, atol=1e-3)
    np.testing.assert_allclose(cy, p.y + p.yllcorner, atol=1e-3)


# ---------------------------------------------------------------------------
# One real short CPU simulation: a pulse BETWEEN two yields
# ---------------------------------------------------------------------------


@pytest.mark.requires_anuga
def test_pulse_between_yields_is_caught_by_the_engine_not_the_frames(tmp_path, monkeypatch):
    """A +0.5 m Dirichlet pulse enters a shallow (0.1 m) flat channel at t in (5, 25) s and
    drains back out; frames are saved at t = 0, 30, 60 s only. The frame max
    misses the pulse at the inflow end; the engine operator (attached through
    run_anuga's own helper, CPU update_frequency=10) catches it, and the
    playback envelope built from it sits above every saved frame."""
    import anuga
    from run_anuga import playback_store as ps

    monkeypatch.delenv(peaks.TRUE_PEAKS_ENV, raising=False)
    domain = anuga.rectangular_cross_domain(50, 4, len1=200.0, len2=20.0)
    domain.set_name("pulse")
    domain.set_datadir(str(tmp_path))
    domain.set_flow_algorithm("DE0")
    domain.set_quantity("elevation", 0.0)
    domain.set_quantity("friction", 0.03)
    domain.set_quantity("stage", 0.1)  # wet, so CFL keeps dt ~1 s
    domain.set_minimum_storable_height(defaults.MINIMUM_STORABLE_HEIGHT_M)

    def pulse(t):
        return [0.6 if 5.0 < t < 25.0 else 0.1, 0.0, 0.0]

    reflective = anuga.Reflective_boundary(domain)
    domain.set_boundary(
        {
            "left": anuga.Time_boundary(domain=domain, function=pulse),
            "right": reflective,
            "top": reflective,
            "bottom": reflective,
        }
    )
    op = peaks.attach_peak_operator(domain, 1)
    assert op is not None
    for _ in domain.evolve(yieldstep=30.0, finaltime=60.0):
        pass

    sww = tmp_path / "pulse.sww"
    engine = peaks.read_engine_peaks(sww)
    assert engine is not None

    import netCDF4

    with netCDF4.Dataset(str(sww)) as ds:
        assert len(ds.variables["time"]) == 3
        stage_c = np.asarray(ds.variables["stage_c"][:], dtype=np.float64)
        elev_c = np.asarray(ds.variables["elevation_c"][:], dtype=np.float64)
    frame_max_depth = np.max(np.maximum(stage_c - elev_c, 0.0), axis=0)
    gain = engine["max_depth_c"] - frame_max_depth
    assert gain.max() > 0.1, f"engine caught no between-yield peak (max gain {gain.max():.4f} m)"

    sww_arrays = ps._read_sww_arrays(sww)
    depth = np.clip(sww_arrays["stage"] - sww_arrays["elevation"][np.newaxis, :], 0, None)
    h0 = 1e-6
    xv = ps.compute_velocity(sww_arrays["xmomentum"], depth, h0)
    yv = ps.compute_velocity(sww_arrays["ymomentum"], depth, h0)
    frames = ps.compute_envelopes(depth, xv, yv)
    n_node = sww_arrays["x"].shape[0]
    env, source = ps.apply_engine_envelopes(frames, sww_arrays["engine_peaks"], sww_arrays["volumes"], n_node)
    assert source == peaks.ENVELOPE_SOURCE_ENGINE
    for name in ps.ENVELOPE_QUANTITIES:
        assert np.all(env[name] >= frames[name])
    # The between-yield pulse shows in the envelope too.
    assert float((env["depth"] - frames["depth"]).max()) > 0.1
    # Every saved frame lies under the envelope.
    assert np.all(env["depth"][np.newaxis, :] >= depth.astype(np.float32) - 1e-6)
