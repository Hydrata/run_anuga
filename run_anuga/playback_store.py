"""SWW -> quantized playback-store exporter (TASK-2622, W1.1, epic 2618).

Implements the schema at
``docs/reports/2026-08-04-task-2619-playback-store-schema.html`` (v1,
operator-signed 2026-08-06, nine gate amendments applied). That document is
LAW for this module's on-disk layout, dtypes, attrs, and quantization math —
read it before changing anything here. In particular (B1): the store is a
REAL **Zarr v3** store (``zarr.json`` group/array metadata, ``c/`` chunk
key prefix, ``/`` separator) — there is no ``.zattrs``/``.zgroup`` anywhere.
zarr-python 3.3.0's own default (``zarr.config.config['default_zarr_format']
== 3``) matches this.

Import-guarded: ``zarr`` is optional (three install surfaces — the
``run_anuga[playback]`` pyproject extra, ``batch/Dockerfile``'s explicit pip
list, and the web venv). A missing zarr degrades the export to a logged
warning, never a failed run — see :func:`export_playback_store`.

NOT touched by this module: the SWW file (read-only) and the ``*_max.tif``
max-envelope rasters themselves (unchanged sibling artifact, cold-archived
separately). TASK-2752 (v2, epic 2706 W8.2) DOES add this module's own
in-browser equivalent of them — ``depth_max``/``velocity_max``/``div_max``
per-vertex arrays, declared via the ``envelope_quantities`` group attr — see
:func:`compute_envelopes`.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# --- Schema constants (§0-§6 of the signed doc) -----------------------------

#: v2 (TASK-2719, epic 2706 W8, decision D5) — the store now declares its own
#: n_node/n_time/chunk_length_t group attrs and the time-chunk length is
#: ADAPTIVE (derive_chunk_length_t) rather than the v1 fixed CHUNK_LENGTH_T.
#: v1 stores (no new attrs, chunk length always 10) remain valid forever —
#: validate_playback_store.py gates the three new required attrs on
#: format_version >= 2.
#: v3 (TASK-2989, epic 2981 W3.1, decision D5) — the three time-series arrays
#: MAY carry the ``temporal_delta`` array->array codec ahead of bytes+gzip.
#: The bump is what tells a client "this store's codec chain is not
#: necessarily bytes+gzip, read the manifest's codecs block"; v1/v2 stores are
#: unaffected and stay valid forever. NOTE the bump belongs to TASK-2989 and
#: NOT to TASK-3014 (the Morton reorder): a reorder is invisible to every
#: reader, a codec chain is not.
FORMAT_VERSION = 3
#: O1 (v1) — ratified at 10, NOT 20: time-blocking buys zero compression
#: (DEFLATE's 32 KiB window cannot span a 1.17 MB timestep row). Chunk length
#: is a pure seek-latency/request-count tradeoff with no byte cost either way.
#: SUPERSEDED for n_node > 838,860 by decision D5 (TASK-2719): the client's
#: per-chunk decode/decompress cost at prod scale (run 1328, n_node
#: 3,393,075) makes chunk 10 the wrong tradeoff — see derive_chunk_length_t.
#: Kept as the CAP (and the exact value every store <= 838,860 nodes still
#: gets, byte-identical to today).
CHUNK_LENGTH_T = 10
#: D5 (TASK-2719) — floor: chunk length 1 recreates the 2618 client-side LRU
#: thrash by construction (one static mesh array, face_node_connectivity, is
#: 1.33x the whole chunk-1 cache ceiling — see the task's Context for the
#: verified arithmetic). 2 is the smallest length that keeps the static
#: arrays inside the cache ceiling at run-1328 scale.
CHUNK_LENGTH_T_FLOOR = 2
#: D5 (TASK-2719) — the per-chunk, per-quantity STORED (uint16, pre-gzip)
#: byte budget the adaptive rule targets. Derived once from the exporter
#: side; gmc's memory-policy constants are a SEPARATE, deliberately
#: un-shared number (cross-repo duplication would let the two drift silently
#: — see the task's "DO NOT copy gmc's memory constants into python" note).
PLAYBACK_TARGET_CHUNK_BYTES = 16 * 1024 * 1024
#: B7 — the solver's own convention (q/(h + h0/h)), NOT plot_utils' masked
#: q/(h+1e-12). The two differ by up to ~5e-2 m/s at the wet/dry fringe.
VELOCITY_CONVENTION = "solver_epsilon"
VELOCITY_FORMULA = "u = uh / (h + h0/h)"
#: §5 — pinned away from Zarr v3's zstd default (browsers had no
#: DecompressionStream('zstd') as of the W0 gate); level pinned explicitly
#: because numcodecs defaults to 1 and zarr v3 to 5.
CODEC_NAME = "gzip"
CODEC_LEVEL = 6
#: Zero code for the symmetric velocity range (B3) — exactly representable.
VELOCITY_FILL_VALUE = 32767
DEPTH_FILL_VALUE = 0
#: TASK-2752 (W8.2, epic 2706) — the temporal-max envelope quantities, matching
#: the `*_max.tif` raster set the batch pipeline already produces (module
#: header above: "NOT touched by this module ... unchanged sibling artifact").
#: Order is deterministic (written to group_attrs verbatim) — do not reorder
#: casually, it is part of the manifest's advertised capability list.
ENVELOPE_QUANTITIES = ("depth", "velocity", "div")
#: Every envelope quantity is a non-negative MAGNITUDE (max depth, max speed,
#: max depth*speed) — quantize_range(0.0, max) always maps code 0 -> physical
#: 0.0, so a single shared fill value works for all three (mirrors
#: DEPTH_FILL_VALUE's reasoning, not VELOCITY_FILL_VALUE's — velocity_max is a
#: magnitude, never signed, so it does NOT get the symmetric [-v,+v] treatment
#: x_velocity/y_velocity use).
ENVELOPE_FILL_VALUE = 0
#: TASK-3014 (W3.0, epic 2981) — the permutation arrays' fill value. -1 is not
#: a legal index into anything, so a chunk that was never written can never be
#: mistaken for "node 0" the way a 0 fill could.
PERMUTATION_FILL_VALUE = -1
#: TASK-2709 (W2.1, epic 2706) — the cache directive written on EVERY playback
#: object at export. S3 cannot add one later without rewriting the object, so
#: it has to be set here or not at all (existing stores can never satisfy it;
#: only a fresh export does).
#:
#: A store is write-once under a per-run prefix
#: (``playback/{project}_{scenario}_{run}/``), so its bytes genuinely are
#: immutable — ``immutable`` is what stops the browser spending a conditional
#: GET round-trip per chunk just to be told 304.
#:
#: The nominal year is NOT how long a stale object can be served: the browser
#: caches against the full presigned URL, and TASK-2710 rotates that URL every
#: time bucket, so the effective ceiling is one bucket, not a year.
PLAYBACK_CACHE_CONTROL = "public, max-age=31536000, immutable"


def zarr_available() -> bool:
    """True if the optional ``zarr`` dependency is importable."""
    try:
        import zarr  # noqa: F401

        return True
    except ImportError:
        return False


def derive_chunk_length_t(n_node: int) -> int:
    """The adaptive time-chunk length for a store with ``n_node`` mesh nodes.

    A PURE function of n_node alone (D5) — same n_node, any n_time, same
    result. ``min(CHUNK_LENGTH_T, max(CHUNK_LENGTH_T_FLOOR, bytes_budget //
    (n_node * 2)))``: 2 stored bytes per (timestep, node) uint16 cell, clamped
    to [2, 10]. n_node <= 838,860 -> 10 (byte-identical to every store
    exported before TASK-2719 — nothing to re-export); n_node >= 2,796,203 ->
    2 (the FLOOR — run 1328's n_node=3,393,075 lands here).
    """
    return min(
        CHUNK_LENGTH_T,
        max(CHUNK_LENGTH_T_FLOOR, PLAYBACK_TARGET_CHUNK_BYTES // (n_node * 2)),
    )


# --- Quantization math (schema §3 "Quantization contract") ------------------


def quantize_range(min_val: float, max_val: float) -> tuple[float, float]:
    """Scale/offset for ``[min_val, max_val]`` with the mandatory degenerate-
    range guard (schema §3): ``scale = (max-min)/65535`` divides by zero when
    max == min — not hypothetical, the SWW's own ``friction_range`` is
    measured ``[0.0400, 0.0400]``, and any fully-dry run gives depth ``[0, 0]``.

    ``raw = offset + stored.astype(float32) * scale``
    ``stored = round((raw - offset) / scale).clip(0, 65535).astype(uint16)``
    """
    min_val = float(min_val)
    max_val = float(max_val)
    if not (np.isfinite(min_val) and np.isfinite(max_val)):
        raise ValueError(f"quantization range is non-finite: [{min_val}, {max_val}]")
    if max_val < min_val:
        raise ValueError(f"quantization range inverted: [{min_val}, {max_val}]")
    if max_val == min_val:
        return 1.0, min_val
    return (max_val - min_val) / 65535.0, min_val


def symmetric_velocity_range(v_absmax: float) -> tuple[float, float]:
    """B3 — symmetric range so zero is exactly representable (zero code
    32767): ``offset = -v_absmax``, ``scale = 2*v_absmax/65534``. Fixes the
    v0 defect where an asymmetric signed range put raw 0.0 off-grid
    (reconstructing to +6.809e-05, a constant directional drift on every
    still/dry cell).
    """
    v_absmax = float(v_absmax)
    if not np.isfinite(v_absmax) or v_absmax < 0:
        raise ValueError(f"v_absmax must be finite and >= 0, got {v_absmax}")
    if v_absmax == 0:
        return 1.0, 0.0
    return 2.0 * v_absmax / 65534.0, -v_absmax


def quantize(raw: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """``stored = round((raw - offset) / scale).clip(0, 65535).astype(uint16)``"""
    stored = np.round((raw.astype(np.float64) - offset) / scale)
    return np.clip(stored, 0, 65535).astype(np.uint16)


# --- Geometry / physics (schema §2-§3) ---------------------------------------


def compute_inradius(x: np.ndarray, y: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """B6 — per-triangle inradius: the MINIMUM centroid-to-edge-midpoint
    distance, matching ANUGA's own ``general_mesh.py`` ``self.radii``. NOT
    area/semiperimeter (the ``use_inscribed_circle=True`` branch — default
    False, never set in this pipeline). On a 3-4-5 triangle the two formulas
    differ 0.8333 vs 1.0 — the wrong one runs client Courant ~17% low on
    skewed cells, the unconservative direction for a risk indicator.
    """
    v0, v1, v2 = volumes[:, 0], volumes[:, 1], volumes[:, 2]
    p0 = np.stack([x[v0], y[v0]], axis=1)
    p1 = np.stack([x[v1], y[v1]], axis=1)
    p2 = np.stack([x[v2], y[v2]], axis=1)
    centroid = (p0 + p1 + p2) / 3.0
    mid01, mid12, mid20 = (p0 + p1) / 2.0, (p1 + p2) / 2.0, (p2 + p0) / 2.0
    d0 = np.linalg.norm(centroid - mid01, axis=1)
    d1 = np.linalg.norm(centroid - mid12, axis=1)
    d2 = np.linalg.norm(centroid - mid20, axis=1)
    return np.minimum(np.minimum(d0, d1), d2).astype(np.float32)


def compute_velocity(momentum: np.ndarray, depth: np.ndarray, h0: float) -> np.ndarray:
    """B7 solver epsilon-form: ``u = uh / (h + h0/h)`` — the value that
    actually advanced the simulation (NOT ``anuga.utilities.plot_utils``'
    masked ``q/(h+1e-12)`` convention). Dry cells (``depth == 0``) are forced
    to exactly 0.0 rather than left to float through a ``0/inf`` division.
    """
    depth = np.asarray(depth, dtype=np.float64)
    momentum = np.asarray(momentum, dtype=np.float64)
    wet = depth > 0
    u = np.zeros_like(depth)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = depth[wet] + h0 / depth[wet]
        u[wet] = momentum[wet] / denom
    return u


def compute_envelopes(
    depth: np.ndarray, x_velocity: np.ndarray, y_velocity: np.ndarray
) -> dict[str, np.ndarray]:
    """TASK-2752 (W8.2, epic 2706) — per-vertex temporal-max envelopes.

    THE TRAP: max-of-derived is NOT derived-of-max. ``velocity`` here is the
    per-timestep speed ``sqrt(x_velocity**2 + y_velocity**2)`` — i.e. it is
    derived FIRST, at every timestep, and ONLY THEN maxed over time. The wrong
    (and cheaper-looking) shortcut — ``sqrt(max(x_velocity)**2 +
    max(y_velocity)**2)`` — silently combines the x- and y-peaks from
    DIFFERENT timesteps into a magnitude neither timestep ever produced,
    which is exactly why ``depth_max``, ``velocity_max`` and ``dIV_max``
    already exist as three separate rasters in the batch pipeline rather than
    being reconstructed from the component maxima.

    ``depth``, ``x_velocity``, ``y_velocity`` are the exporter's already-
    physical (post ``compute_velocity``) ``[n_time, n_node]`` arrays — the
    SAME arrays quantize_range/quantize are then run over for the primitive
    depth/x_velocity/y_velocity arrays, so this function must run BEFORE
    those are overwritten by their quantized (uint16) counterparts.

    @returns dict with keys :data:`ENVELOPE_QUANTITIES` ('depth', 'velocity',
    'div'), each a ``(n_node,)`` float32 array, all >= 0 by construction.
    """
    depth = np.asarray(depth, dtype=np.float64)
    x_velocity = np.asarray(x_velocity, dtype=np.float64)
    y_velocity = np.asarray(y_velocity, dtype=np.float64)
    if depth.size == 0:
        n_node = depth.shape[1] if depth.ndim == 2 else 0
        zeros = np.zeros(n_node, dtype=np.float32)
        return {name: zeros.copy() for name in ENVELOPE_QUANTITIES}

    depth_clamped = np.maximum(depth, 0.0)
    # DERIVED FIRST, at every (t, node) — the speed a viewer would actually
    # see rendered at that instant — THEN maxed over t. Never the other order.
    speed = np.sqrt(np.square(x_velocity) + np.square(y_velocity))
    div = depth_clamped * speed

    return {
        "depth": np.max(depth_clamped, axis=0).astype(np.float32),
        "velocity": np.max(speed, axis=0).astype(np.float32),
        "div": np.max(div, axis=0).astype(np.float32),
    }


# --- SWW reading --------------------------------------------------------------


def _read_sww_arrays(sww_path) -> dict[str, Any]:
    """Read the raw arrays + global attrs this exporter needs from an SWW.

    Uses netCDF4 directly (an existing run_anuga dep, batch/Dockerfile:43) —
    the same NetCDF file ANUGA's own ``anuga.file.sww`` writer produces.
    Raises if ``smoothing`` is not exactly ``'Yes'`` (schema §5 provenance
    rule — asserted, not assumed, so a non-smoothed SWW cannot be silently
    mislabelled as the vertex-averaged layout this store's naming/geometry
    contract assumes).
    """
    import netCDF4

    with netCDF4.Dataset(str(sww_path)) as ds:
        smoothing = getattr(ds, "smoothing", None)
        if smoothing != "Yes":
            raise ValueError(
                f"playback_store requires a smoothed (vertex-averaged) SWW; "
                f"got smoothing={smoothing!r} (schema §5 provenance rule)."
            )
        return dict(
            x=np.array(ds.variables["x"][:], dtype=np.float32),
            y=np.array(ds.variables["y"][:], dtype=np.float32),
            volumes=np.array(ds.variables["volumes"][:], dtype=np.int32),
            elevation=np.array(ds.variables["elevation"][:], dtype=np.float32),
            friction=np.array(ds.variables["friction"][:], dtype=np.float32),
            time=np.array(ds.variables["time"][:], dtype=np.float64),
            stage=np.array(ds.variables["stage"][:], dtype=np.float32),
            xmomentum=np.array(ds.variables["xmomentum"][:], dtype=np.float32),
            ymomentum=np.array(ds.variables["ymomentum"][:], dtype=np.float32),
            xllcorner=float(getattr(ds, "xllcorner", 0.0)),
            yllcorner=float(getattr(ds, "yllcorner", 0.0)),
            false_easting=float(getattr(ds, "false_easting", 0.0)),
            false_northing=float(getattr(ds, "false_northing", 0.0)),
            zone=int(getattr(ds, "zone", -1)),
            anuga_version=str(getattr(ds, "anuga_version", "0.0.0+unknown")),
            revision_number=str(getattr(ds, "revision_number", "")),
            revision_date=str(getattr(ds, "revision_date", "")),
        )


def _load_dt_ms_series(output_dir, run_label, n_time: int):
    """O2 — best-effort dt_ms(t) from the run's diagnostics CSV(s).

    §8 (non-blocking, NOT decided) leaves the dt_ms[0]-is-always-invalid
    question and the CSV/SWW row-count-mismatch behaviour open — this
    function does NOT invent a resolution for either: it applies only the
    two positional cases that need no judgement call (exact match; CSV one
    short because it starts after the SWW's t=0 sample), and degrades to
    has_dt=False otherwise. Prefers min_dt_ms (TASK-2622's new column) over
    mean_dt_ms for older CSVs that predate it.
    """
    import csv
    import glob

    pattern = str(Path(output_dir) / "run_diagnostics_*.csv")
    matches = sorted(glob.glob(pattern))
    # A specific batch-numbered file matching this run_label's batch, if the
    # caller can't disambiguate multi-batch runs, just take the first —
    # dt_ms is advisory display data, not correctness-critical.
    if not matches:
        return np.full(n_time, np.nan, dtype=np.float32), False, None

    rows = []
    try:
        with open(matches[0], newline="") as f:
            reader = csv.DictReader(line for line in f if not line.startswith("#"))
            rows = list(reader)
    except Exception:
        logger.warning("playback_store: failed reading %s for dt_ms", matches[0], exc_info=True)
        return np.full(n_time, np.nan, dtype=np.float32), False, None

    if not rows:
        return np.full(n_time, np.nan, dtype=np.float32), False, None

    dt_source = "min_dt_ms" if "min_dt_ms" in rows[0] else (
        "mean_dt_ms" if "mean_dt_ms" in rows[0] else None
    )
    if dt_source is None:
        return np.full(n_time, np.nan, dtype=np.float32), False, None

    try:
        col = np.array([float(r[dt_source]) for r in rows], dtype=np.float32)
    except (KeyError, ValueError):
        return np.full(n_time, np.nan, dtype=np.float32), False, None

    if len(col) == n_time:
        return col, True, dt_source
    if len(col) == n_time - 1:
        # CSV starts after the SWW's initial (t=0) sample — schema §8 flags
        # dt_ms[0] as "always invalid" when skip_initial_step=False; NaN is
        # the conservative option (never fabricate a number for it).
        return np.concatenate([[np.nan], col]).astype(np.float32), True, dt_source
    logger.warning(
        "playback_store: diagnostics CSV row count (%d) doesn't align with "
        "SWW timesteps (%d) — has_dt=False (schema §8, unresolved).",
        len(col), n_time,
    )
    return np.full(n_time, np.nan, dtype=np.float32), False, None


# --- Zarr v3 store writer -----------------------------------------------------


def _write_zarr_v3_store(store_path, *, group_attrs: dict, arrays: dict):
    """Write one Zarr v3 group with the arrays in ``arrays``.

    ``arrays`` maps array name -> dict(data=ndarray, chunks=tuple,
    fill_value=..., attrs=dict). Every array gets the schema-pinned codec
    chain + chunk_key_encoding (schema §1).
    """
    import zarr
    from zarr.codecs import BytesCodec, GzipCodec

    root = zarr.open_group(str(store_path), mode="w", zarr_format=3)
    for key, value in group_attrs.items():
        root.attrs[key] = value

    for name, spec in arrays.items():
        data = spec["data"]
        arr = root.create_array(
            name,
            shape=data.shape,
            chunks=spec["chunks"],
            dtype=data.dtype,
            # TASK-2989 — array->array filters, written BEFORE the serializer
            # in the zarr v3 `codecs` list. Defaults to none, so every array
            # that does not ask for one is byte-identical to before.
            filters=spec.get("filters", []),
            serializer=BytesCodec(endian="little"),
            compressors=GzipCodec(level=CODEC_LEVEL),
            fill_value=spec["fill_value"],
            chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        )
        arr[:] = data
        for k, v in spec.get("attrs", {}).items():
            arr.attrs[k] = v


def _upload_store_to_s3(store_path, bucket: str, prefix: str) -> str:
    """Upload every file under ``store_path`` to ``s3://bucket/prefix``,
    preserving the store's relative directory structure (mirrors the
    multipart-config pattern in ``_handoff.py`` upload_cold_archive).

    Every object is written with ``PLAYBACK_CACHE_CONTROL`` (TASK-2709) —
    metadata and chunks alike, since the browser fetches both through the same
    presigned manifest URLs and re-downloading 62.7 MiB of geometry per page
    load is exactly the cost this removes.
    """
    import boto3
    from boto3.s3.transfer import TransferConfig

    s3 = boto3.client("s3")
    transfer_config = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,
        multipart_chunksize=8 * 1024 * 1024,
    )
    store_path = Path(store_path)
    n_files = 0
    for local_file in store_path.rglob("*"):
        if not local_file.is_file():
            continue
        rel = local_file.relative_to(store_path).as_posix()
        key = f"{prefix}{rel}"
        s3.upload_file(
            str(local_file), bucket, key,
            ExtraArgs={"CacheControl": PLAYBACK_CACHE_CONTROL},
            Config=transfer_config,
        )
        n_files += 1
    logger.info(
        "playback_store: uploaded %d files to s3://%s/%s", n_files, bucket, prefix
    )
    return prefix


def make_playback_store_prefix(project_id, scenario_id, run_id) -> str:
    """Canonical S3 prefix for a run's playback store (schema §1).

    Mirrors ``_handoff.make_cold_archive_prefix``'s ``{project}_{scenario}_
    {run}/`` shape under its own ``playback/`` root (NOT cold-archive/ —
    different bucket lifecycle policy).
    """
    return f"playback/{project_id}_{scenario_id}_{run_id}/"


# --- Cross-process result handoff (TASK-2623, W1.2, epic 2618) --------------
#
# export_playback_store() runs deep inside run_sim() (post_process_sww's
# rank-0 PHASE_COG_EXPORT seam, run.py:713-725) — several call frames below
# _handoff.run_and_report(), which is where report_result() posts back to
# the Hydrata control server. run_and_report() has no other channel to learn
# whether the export actually uploaded a store (vs skipped/errored) — unlike
# the bucket (RESULT_S3_BUCKET, backfilled into os.environ), a boolean
# outcome can't be "backfilled" before the fact. A small JSON marker file in
# the SAME output_dir both functions already agree on (see
# _handoff.upload_cold_archive's identical output_dir formula) is the
# simplest reliable channel — written as this module's last step on every
# code path (ok/skipped/error), read back by _handoff.run_and_report.

_MARKER_FILENAME = "playback_store_export_result.json"


def _write_marker(output_dir, result: dict) -> None:
    import json

    try:
        marker_path = Path(output_dir) / _MARKER_FILENAME
        marker_path.write_text(json.dumps(result))
    except Exception:
        logger.warning("playback_store: failed writing %s", _MARKER_FILENAME, exc_info=True)


def read_playback_store_marker(output_dir) -> dict | None:
    """Read back the marker written by export_playback_store(), or None if
    absent/unreadable (e.g. this run predates TASK-2622, or export never ran).
    """
    import json

    marker_path = Path(output_dir) / _MARKER_FILENAME
    if not marker_path.is_file():
        return None
    try:
        return json.loads(marker_path.read_text())
    except Exception:
        logger.warning("playback_store: failed reading %s", _MARKER_FILENAME, exc_info=True)
        return None


def playback_store_prefix_for_run(package_dir, project_id, scenario_id, run_id) -> str | None:
    """Convenience wrapper for _handoff.run_and_report: the S3 prefix to
    report back to the BE, or None when the export didn't produce an
    uploaded store (skipped/errored/local-only/marker missing).
    """
    output_dir = Path(package_dir) / f"outputs_{project_id}_{scenario_id}_{run_id}"
    marker = read_playback_store_marker(output_dir)
    if marker is None or marker.get("status") != "ok":
        return None
    return marker.get("s3_prefix")


# --- Orchestration -------------------------------------------------------------


def export_playback_store(
    *,
    input_data: dict,
    sww_path,
    output_dir,
    domain=None,
    upload: bool = True,
    bucket: str | None = None,
    prefix: str | None = None,
    spatial_order: bool = True,
    temporal_delta: bool = True,
) -> dict:
    """Export ``sww_path`` to a local Zarr v3 playback store under
    ``output_dir``, then (if ``upload``) push it to S3.

    Never raises — a missing ``zarr`` or any export failure degrades to a
    logged warning and ``{"status": ...}``, so the run itself never fails
    (landmine #1, TASK-2622 context). Returns a dict with at least a
    ``status`` key: ``"ok"``, ``"skipped_no_zarr"``, or ``"error"``.

    ``spatial_order`` (TASK-3014, W3.0, epic 2981) writes the mesh in Morton /
    Z-order — see :func:`_apply_spatial_order`. It defaults ON because it is
    client-transparent and buys 1.30x on the blocking geometry prefix; passing
    ``False`` writes exactly the store this exporter wrote before TASK-3014
    (SWW order, no permutation arrays, no order attrs), which is what makes
    the reorder's own byte case falsifiable.

    ``temporal_delta`` (TASK-2989, W3.1) allows the ``temporal_delta`` codec on
    the three time-series arrays. It is a PERMISSION, not a command: the codec
    is applied only when ``derive_chunk_length_t(n_node) >= 5`` (decision D5 —
    below that it measured 96% on depth and LARGER on velocity). Passing
    ``False`` switches it off unconditionally, which is what makes "a store
    below the gate is byte-identical to a pre-TASK-2989 export" testable.
    """
    if not zarr_available():
        logger.warning(
            'playback_store: zarr not installed — skipping playback-store '
            'export (pip install "run_anuga[playback]" to enable). Run '
            'continues normally; per-timestep TIFs / *_max.tif are unaffected.'
        )
        result = {"status": "skipped_no_zarr"}
        _write_marker(output_dir, result)
        return result

    try:
        result = _export_playback_store_impl(
            input_data=input_data,
            sww_path=sww_path,
            output_dir=output_dir,
            domain=domain,
            upload=upload,
            bucket=bucket,
            prefix=prefix,
            spatial_order=spatial_order,
            temporal_delta=temporal_delta,
        )
    except Exception:
        logger.exception(
            "playback_store: export failed; run continues without a playback store"
        )
        result = {"status": "error"}
    _write_marker(output_dir, result)
    return result


def _apply_spatial_order(*, node_x, node_y, volumes):
    """TASK-3014 (W3.0, epic 2981) — the Morton reorder, as one pure step.

    @returns ``(node_perm, face_perm, remapped_volumes)``. Both permutations
    are ``int64`` and genuine (``perm[newIndex] == originalIndex``);
    ``remapped_volumes`` is ``volumes`` with its VALUES already substituted
    through the inverse node permutation, still in the ORIGINAL face order.

    The remapped volumes come back rather than being recomputed by the caller
    because this function has to build them anyway (a face's Morton key is its
    centroid's, which needs the new node coordinates) and at run-1328 scale
    that array is 6,779,432 x 3 int64 = 163 MB. Computing it twice allocated it
    twice, on the one code path that already runs at the memory ceiling.

    WHY FACES MOVE TOO, and why that is not optional. Measured on the two real
    fixtures under the store's own bytes+gzip(6) chain, reorder only (no delta,
    no byte-shuffle — those belong to W3.1/W3.2/W3.3):

        variant             face_node_connectivity          client prefix
        nodes only          35,435,624 -> 32,615,509 (1.09x)     1.10x
        nodes AND faces     35,435,624 -> 23,552,802 (1.50x)     1.30x

    A node reorder alone leaves the three integers in a connectivity ROW
    spatially coherent but leaves the ROWS in mesh-generation order, so
    consecutive rows still jump across the mesh and DEFLATE's 32 KiB window
    finds nothing. Sorting the rows by their centroid's Z-key is what makes the
    high bytes of consecutive rows repeat. Nodes-only buys 1.09x on
    connectivity, i.e. below TASK-3014 AC2's 1.3x escalation threshold; nodes
    plus faces buys 1.50x. Same shape on the small store (1.54x).

    TRIANGLE WINDING IS PRESERVED: this reorders ROWS and substitutes VALUES,
    and never sorts the three vertices inside a row. The critique measured the
    orientation-preserving variant explicitly (9,221,596 B vs 9,094,123 B for
    the vertex-sorted one) and the renderer depends on the orientation, so the
    127 KB is deliberately left on the table.
    """
    from run_anuga.playback_order import inverse_permutation, morton_node_order

    node_perm = morton_node_order(node_x, node_y)
    inv = inverse_permutation(node_perm)
    # Face keys are computed in the NEW node order so the two curves agree;
    # the centroid is the same point either way, this just avoids a second
    # gather over the original coordinates.
    new_x = node_x[node_perm]
    new_y = node_y[node_perm]
    remapped = inv[volumes]
    face_perm = morton_node_order(
        new_x[remapped].mean(axis=1), new_y[remapped].mean(axis=1)
    )
    return node_perm, face_perm, remapped


def _export_playback_store_impl(
    *, input_data, sww_path, output_dir, domain, upload, bucket, prefix,
    spatial_order: bool = True, temporal_delta: bool = True,
) -> dict:
    from run_anuga import defaults

    scenario_config = input_data["scenario_config"]
    run_label = input_data["run_label"]

    sww = _read_sww_arrays(sww_path)
    n_time, n_node = sww["stage"].shape

    depth = np.clip(sww["stage"] - sww["elevation"][np.newaxis, :], 0, None).astype(np.float32)

    import anuga.config as anuga_config

    h0 = float(getattr(anuga_config, "velocity_protection", 1e-6))
    g = float(getattr(anuga_config, "g", 9.8))
    rho_w = float(getattr(anuga_config, "rho_w", 1023))

    if domain is not None:
        minimum_allowed_height = float(domain.minimum_allowed_height)
        flow_algorithm = str(getattr(domain, "flow_algorithm", scenario_config.get("flow_algorithm", "DE0")))
    else:
        # schema §5 wants "read from the run's domain at export" — no live
        # domain (e.g. CLI re-export from an archived SWW) degrades to the
        # anuga.config module default + scenario_config, logged so the gap
        # is visible rather than silently passed off as authoritative.
        minimum_allowed_height = float(getattr(anuga_config, "minimum_allowed_height", 1e-5))
        flow_algorithm = str(scenario_config.get("flow_algorithm", "DE0"))
        logger.warning(
            "playback_store: no live domain passed — minimum_allowed_height "
            "read from anuga.config default (%.3g), not the run's actual "
            "domain (schema §5).", minimum_allowed_height,
        )

    x_velocity = compute_velocity(sww["xmomentum"], depth, h0).astype(np.float32)
    y_velocity = compute_velocity(sww["ymomentum"], depth, h0).astype(np.float32)
    inradius = compute_inradius(sww["x"], sww["y"], sww["volumes"])

    dt_ms, has_dt, dt_source = _load_dt_ms_series(output_dir, run_label, n_time)

    # TASK-2752 — computed from the same physical (pre-quantization)
    # depth/x_velocity/y_velocity arrays quantize_range/quantize are about to
    # consume below. Must run before those names are shadowed by their
    # quantized (uint16) counterparts a few lines down.
    envelopes = compute_envelopes(depth, x_velocity, y_velocity)

    max_depth = float(np.max(depth)) if depth.size else 0.0
    depth_scale, depth_offset = quantize_range(0.0, max_depth)
    depth_q = quantize(depth, depth_scale, depth_offset)

    v_absmax = float(
        max(
            np.max(np.abs(x_velocity)) if x_velocity.size else 0.0,
            np.max(np.abs(y_velocity)) if y_velocity.size else 0.0,
        )
    )
    v_scale, v_offset = symmetric_velocity_range(v_absmax)
    x_velocity_q = quantize(x_velocity, v_scale, v_offset)
    y_velocity_q = quantize(y_velocity, v_scale, v_offset)

    # TASK-2752 — each envelope gets its OWN [0, max] quantization range,
    # independent of the primitive arrays' ranges: velocity_max (a speed
    # MAGNITUDE, max sqrt(vx^2+vy^2)) is never smaller than v_absmax (the
    # componentwise |vx|/|vy| max symmetric_velocity_range above is keyed on)
    # and is frequently larger, so sharing v_scale/v_offset would silently
    # clip the envelope's own peak.
    envelope_quantized = {}
    envelope_attrs = {}
    for name in ENVELOPE_QUANTITIES:
        raw = envelopes[name]
        env_max = float(np.max(raw)) if raw.size else 0.0
        env_scale, env_offset = quantize_range(0.0, env_max)
        envelope_quantized[name] = quantize(raw, env_scale, env_offset)
        envelope_attrs[name] = dict(
            scale=env_scale, offset=env_offset, quantized_dtype="uint16",
            byteorder="little", valid_min=0.0, valid_max=env_max,
        )

    # TASK-3014 (W3.0, epic 2981) — THE SPATIAL REORDER, applied to every
    # per-node axis and to the connectivity's VALUES, immediately before the
    # arrays dict is built and after every quantization range has been taken.
    #
    # ORDER-INVARIANCE IS WHY THIS SITS HERE. quantize_range/symmetric_
    # velocity_range are reductions over the WHOLE array (min/max/absmax), so
    # they give the identical scale/offset whichever order the nodes are in —
    # which means permuting the already-QUANTIZED uint16 arrays is both
    # bit-identical to permuting the physical float32 ones and half the memory
    # traffic. compute_envelopes and compute_inradius likewise ran above, in
    # SWW order, and their results are simply gathered here.
    node_x = sww["x"]
    node_y = sww["y"]
    elevation = sww["elevation"]
    friction = sww["friction"]
    volumes = sww["volumes"]
    node_permutation = None
    face_permutation = None
    if spatial_order:
        from run_anuga.playback_order import MORTON_ORDER_NAME

        node_permutation, face_permutation, remapped_volumes = _apply_spatial_order(
            node_x=node_x, node_y=node_y, volumes=volumes,
        )
        node_x = node_x[node_permutation]
        node_y = node_y[node_permutation]
        elevation = elevation[node_permutation]
        friction = friction[node_permutation]
        # VALUES remapped (each row still holds its own three vertices in the
        # same rotational order), then ROWS reordered. Never a sort within a
        # row — winding is load-bearing for the renderer.
        volumes = remapped_volumes.astype(np.int32)[face_permutation]
        inradius = inradius[face_permutation]
        depth_q = depth_q[:, node_permutation]
        x_velocity_q = x_velocity_q[:, node_permutation]
        y_velocity_q = y_velocity_q[:, node_permutation]
        for name in ENVELOPE_QUANTITIES:
            envelope_quantized[name] = envelope_quantized[name][node_permutation]

    epsg = scenario_config.get("epsg")
    model_start = scenario_config.get("model_start", "1970-01-01T00:00:00+00:00")

    chunk_length_t = derive_chunk_length_t(n_node)

    # TASK-2989 (W3.1, epic 2981) — decision D5. The delta pays only where
    # there are enough rows in a chunk for inter-row similarity to survive
    # gzip's window: measured 35%/48% at length 10 on run 1412, but 96% on
    # depth and LARGER on x_velocity at length 2. `temporal_delta=False` is
    # the unconditional off switch; the length rule is the automatic one.
    time_series_filters: list = []
    apply_temporal_delta = False
    if temporal_delta:
        from run_anuga.playback_codecs import (TEMPORAL_DELTA_MIN_CHUNK_LENGTH,
                                               TemporalDeltaCodec)

        apply_temporal_delta = chunk_length_t >= TEMPORAL_DELTA_MIN_CHUNK_LENGTH
        if apply_temporal_delta:
            time_series_filters = [TemporalDeltaCodec()]

    group_attrs = dict(
        format_version=FORMAT_VERSION,
        xllcorner=sww["xllcorner"],
        yllcorner=sww["yllcorner"],
        false_easting=sww["false_easting"],
        false_northing=sww["false_northing"],
        epsg=epsg,
        zone=sww["zone"],
        velocity_convention=VELOCITY_CONVENTION,
        velocity_formula=VELOCITY_FORMULA,
        velocity_protection=h0,
        minimum_allowed_height=minimum_allowed_height,
        display_mask_h=float(defaults.MIN_ALLOWED_HEIGHT_M),
        minimum_storable_height=float(defaults.MINIMUM_STORABLE_HEIGHT_M),
        g=g,
        rho_w=rho_w,
        building_mannings_n=float(defaults.BUILDING_MANNINGS_N),
        flow_algorithm=flow_algorithm,
        model_start=model_start,
        time_units="seconds",
        has_dt=bool(has_dt),
        dt_source=dt_source,
        smoothing="vertex-averaged",
        anuga_version=sww["anuga_version"],
        revision_number=sww["revision_number"],
        revision_date=sww["revision_date"],
        codec=CODEC_NAME,
        codec_level=CODEC_LEVEL,
        # TASK-2719 (v2) — the store declares its own dimensions and the
        # adaptive rule that produced chunk_length_t, rather than making a
        # future reader re-derive them from the chunk grid.
        n_node=int(n_node),
        n_time=int(n_time),
        chunk_length_t=int(chunk_length_t),
        # TASK-2752 (v2, epic 2706 W8.2) — first-class-absence capability
        # flag, the SAME shape has_dt already uses (schema §5): which
        # temporal-max envelopes THIS store actually contains. A store
        # exported before this task simply never declares the key (manifest
        # relay + validator both treat that as "declares none" — no backfill,
        # no 404, no throw; TASK-2752 AC4).
        envelope_quantities=list(ENVELOPE_QUANTITIES),
        # TASK-2989 (v3, W3.1, epic 2981) — decision D5's verdict for THIS
        # store, recorded rather than left to be re-derived. It is written
        # unconditionally at v3 (not presence-gated like node_order) precisely
        # so the validator can catch a store that CLAIMS a codec it does not
        # carry, or carries one it does not claim; the chunk-length rule alone
        # could not tell those apart from a correct store.
        temporal_delta_applied=bool(apply_temporal_delta),
    )
    if spatial_order:
        # TASK-3014 — first-class ABSENCE, the shape has_dt and
        # envelope_quantities already use: a store that never declares these
        # is in the original SWW order and stays valid forever. Declaring the
        # order is what makes node_permutation's presence REQUIRED, so the
        # validator can tell a real permutation from a corrupted one.
        group_attrs["node_order"] = MORTON_ORDER_NAME
        group_attrs["face_order"] = MORTON_ORDER_NAME

    t_chunks = (chunk_length_t, n_node)
    arrays = {
        "node_x": dict(data=node_x, chunks=(n_node,), fill_value=0.0),
        "node_y": dict(data=node_y, chunks=(n_node,), fill_value=0.0),
        "face_node_connectivity": dict(
            data=volumes, chunks=volumes.shape, fill_value=-1
        ),
        "elevation": dict(data=elevation, chunks=(n_node,), fill_value=0.0),
        "friction": dict(data=friction, chunks=(n_node,), fill_value=0.0),
        "inradius": dict(data=inradius, chunks=(inradius.shape[0],), fill_value=0.0),
        "time": dict(data=sww["time"], chunks=(n_time,), fill_value=0.0),
        "dt_ms": dict(data=dt_ms, chunks=(n_time,), fill_value=float("nan")),
        "depth": dict(
            data=depth_q,
            chunks=t_chunks,
            fill_value=DEPTH_FILL_VALUE,
            filters=time_series_filters,
            attrs=dict(
                scale=depth_scale, offset=depth_offset, quantized_dtype="uint16",
                byteorder="little", valid_min=0.0, valid_max=max_depth,
            ),
        ),
        "x_velocity": dict(
            data=x_velocity_q,
            chunks=t_chunks,
            fill_value=VELOCITY_FILL_VALUE,
            filters=time_series_filters,
            attrs=dict(
                scale=v_scale, offset=v_offset, quantized_dtype="uint16",
                byteorder="little", valid_min=-v_absmax, valid_max=v_absmax,
            ),
        ),
        "y_velocity": dict(
            data=y_velocity_q,
            chunks=t_chunks,
            fill_value=VELOCITY_FILL_VALUE,
            filters=time_series_filters,
            attrs=dict(
                scale=v_scale, offset=v_offset, quantized_dtype="uint16",
                byteorder="little", valid_min=-v_absmax, valid_max=v_absmax,
            ),
        ),
    }
    # TASK-2752 — one (n_node,) array per declared envelope quantity, e.g.
    # 'depth' -> 'depth_max'. Single node-chunk, same shape/codec/chunk-key
    # pattern as elevation/friction/inradius (schema §2's static arrays), NOT
    # time-chunked like depth/x_velocity/y_velocity — an envelope has no time
    # axis left to chunk.
    for name in ENVELOPE_QUANTITIES:
        arrays[f"{name}_max"] = dict(
            data=envelope_quantized[name],
            chunks=(n_node,),
            fill_value=ENVELOPE_FILL_VALUE,
            attrs=envelope_attrs[name],
        )
    if spatial_order:
        # TASK-3014 — the only way back to SWW order. THE CLIENT NEVER FETCHES
        # THESE: playbackEpics.js's mesh object list is node_x, node_y,
        # elevation, friction, inradius, face_node_connectivity, time, dt_ms,
        # so they cost S3 storage and nothing at all on the blocking prefix
        # this task exists to shrink. Measured cost, gzip-6, run 1328:
        # node_permutation 6,738,192 B and face_permutation 14,127,682 B —
        # about 50% of raw, i.e. NOT "compresses well", so the trade is a
        # 14.7 MB smaller download against a 20.9 MB bigger object store.
        arrays["node_permutation"] = dict(
            data=node_permutation.astype(np.int32),
            chunks=(n_node,),
            fill_value=PERMUTATION_FILL_VALUE,
        )
        arrays["face_permutation"] = dict(
            data=face_permutation.astype(np.int32),
            chunks=(face_permutation.shape[0],),
            fill_value=PERMUTATION_FILL_VALUE,
        )

    store_path = Path(output_dir) / f"{run_label}_playback.zarr"
    _write_zarr_v3_store(store_path, group_attrs=group_attrs, arrays=arrays)

    result = {
        "status": "ok",
        "local_path": str(store_path),
        "n_time": int(n_time),
        "n_node": int(n_node),
        "has_dt": bool(has_dt),
    }

    if upload:
        resolved_bucket = bucket or os.environ.get("RESULT_S3_BUCKET")
        if not resolved_bucket:
            logger.warning(
                "playback_store: upload requested but no bucket resolved "
                "(pass bucket= or set RESULT_S3_BUCKET) — store written "
                "locally only at %s", store_path,
            )
            result["status"] = "ok_local_only"
            return result
        resolved_prefix = prefix or make_playback_store_prefix(
            scenario_config.get("project"), scenario_config.get("id"), scenario_config.get("run_id")
        )
        _upload_store_to_s3(store_path, resolved_bucket, resolved_prefix)
        result["s3_bucket"] = resolved_bucket
        result["s3_prefix"] = resolved_prefix

    return result
