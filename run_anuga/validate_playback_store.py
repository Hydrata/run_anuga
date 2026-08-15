"""Array-level validator for a playback store (TASK-2622, W1.1, epic 2618).

Checks a local Zarr v3 store directory against the signed schema
(``docs/reports/2026-08-04-task-2619-playback-store-schema.html``, v1):
required arrays, dtypes, shapes, codecs, chunk-key encoding, fill_value, and
the per-array quantization attrs (schema §3 "Quantization contract",
including the degenerate-range guard).

Usage::

    python -m run_anuga.validate_playback_store /path/to/run_label_playback.zarr

Import-guarded on ``zarr`` like the exporter — this tool is only useful when
zarr is installed, but importing the module itself never raises.
"""
from __future__ import annotations

import sys
from pathlib import Path

# name -> (expected dtype kind+itemsize via numpy dtype string, expected ndim)
# Node-shaped static geometry — (nNode,). NOT inradius, which is per-FACE
# (schema §2: "Per-triangle, matching the solver's per-cell self.radii").
_STATIC_ARRAYS = {
    "node_x": ("float32", 1),
    "node_y": ("float32", 1),
    "elevation": ("float32", 1),
    "friction": ("float32", 1),
}
_FACE_ARRAY = "face_node_connectivity"  # int32, (nFace, 3)
_INRADIUS_ARRAY = "inradius"  # float32, (nFace,) — per-triangle, NOT per-node
_TIME_ARRAYS = {
    "time": ("float64", 1),
    "dt_ms": ("float32", 1),
}
_PRIMITIVE_ARRAYS = {
    "depth": 0,
    "x_velocity": 32767,
    "y_velocity": 32767,
}
_REQUIRED_GROUP_ATTRS = [
    "format_version", "xllcorner", "yllcorner", "false_easting", "false_northing",
    "epsg", "zone", "velocity_convention", "velocity_formula", "velocity_protection",
    "minimum_allowed_height", "display_mask_h", "minimum_storable_height", "g", "rho_w",
    "building_mannings_n", "flow_algorithm", "model_start", "time_units", "has_dt",
    "dt_source", "smoothing", "anuga_version", "revision_number", "revision_date",
    "codec", "codec_level",
]
#: TASK-2719 (v2, epic 2706 W8) — n_node/n_time/chunk_length_t are required
#: ONLY for format_version >= 2. A v1 store (chunk length always 10, none of
#: these attrs present — including every store exported before this task,
#: e.g. run 1328's) must keep validating with ZERO violations forever.
_REQUIRED_GROUP_ATTRS_V2 = ["n_node", "n_time", "chunk_length_t"]
_CHUNK_LENGTH_T_FLOOR = 2
_CHUNK_LENGTH_T_CAP = 10
_REQUIRED_QUANT_ATTRS = ["scale", "offset", "quantized_dtype", "byteorder", "valid_min", "valid_max"]
_EXPECTED_CODECS = [
    {"name": "bytes", "configuration": {"endian": "little"}},
    {"name": "gzip", "configuration": {"level": 6}},
]
_EXPECTED_CHUNK_KEY_ENCODING = {"name": "default", "configuration": {"separator": "/"}}


def validate_store(store_path) -> list[str]:
    """Return a list of violation strings; empty list == schema-conformant.

    Never raises for a well-formed but non-conformant store (violations are
    returned, not thrown); a structurally-broken store (unreadable zarr.json,
    missing group) raises via zarr's own errors — the caller is expected to
    have already confirmed ``store_path`` looks like a zarr store.
    """
    import zarr

    violations: list[str] = []
    root = zarr.open_group(str(store_path), mode="r")

    if root.metadata.zarr_format != 3:
        violations.append(f"zarr_format={root.metadata.zarr_format!r}, expected 3 (schema B1)")

    for attr in _REQUIRED_GROUP_ATTRS:
        if attr not in root.attrs:
            violations.append(f"group attrs missing '{attr}' (schema §5)")

    format_version = root.attrs.get("format_version")
    if isinstance(format_version, (int, float)) and format_version >= 2:
        for attr in _REQUIRED_GROUP_ATTRS_V2:
            if attr not in root.attrs:
                violations.append(
                    f"group attrs missing '{attr}' (schema §5, required for format_version >= 2)"
                )

    # TASK-2719 (v2) — the store's OWN declared chunk_length_t is the law
    # for its three quantized arrays' time-chunk length (O1/D5: "clients
    # MUST read them from zarr.json and MUST NOT hardcode them" applies to
    # this validator too). A v1 store never declares it, so O1's original
    # fixed-10 rule still applies unconditionally there.
    declared_chunk_length_t = root.attrs.get("chunk_length_t")
    if declared_chunk_length_t is not None and not (
        _CHUNK_LENGTH_T_FLOOR <= declared_chunk_length_t <= _CHUNK_LENGTH_T_CAP
    ):
        violations.append(
            f"chunk_length_t={declared_chunk_length_t!r}, outside "
            f"[{_CHUNK_LENGTH_T_FLOOR}, {_CHUNK_LENGTH_T_CAP}] (D5)"
        )

    names = set(root.array_keys())
    n_node = None
    n_time = None
    n_face = None

    for name, (dtype_str, ndim) in _STATIC_ARRAYS.items():
        if name not in names:
            violations.append(f"missing required array '{name}' (schema §2)")
            continue
        arr = root[name]
        if str(arr.dtype) != dtype_str:
            violations.append(f"{name}: dtype={arr.dtype}, expected {dtype_str} (schema §2)")
        if arr.ndim != ndim:
            violations.append(f"{name}: ndim={arr.ndim}, expected {ndim}")
        if n_node is None:
            n_node = arr.shape[0]
        elif arr.shape[0] != n_node:
            violations.append(f"{name}: shape[0]={arr.shape[0]} != nNode={n_node}")

    if _FACE_ARRAY not in names:
        violations.append(f"missing required array '{_FACE_ARRAY}' (schema §2)")
    else:
        arr = root[_FACE_ARRAY]
        if str(arr.dtype) != "int32":
            violations.append(f"{_FACE_ARRAY}: dtype={arr.dtype}, expected int32")
        if arr.ndim != 2 or arr.shape[1] != 3:
            violations.append(f"{_FACE_ARRAY}: shape={arr.shape}, expected (nFace, 3)")
        n_face = arr.shape[0]

    for name, (dtype_str, ndim) in _TIME_ARRAYS.items():
        if name not in names:
            violations.append(f"missing required array '{name}' (schema §1)")
            continue
        arr = root[name]
        if str(arr.dtype) != dtype_str:
            violations.append(f"{name}: dtype={arr.dtype}, expected {dtype_str}")
        if n_time is None:
            n_time = arr.shape[0]
        elif arr.shape[0] != n_time:
            violations.append(f"{name}: shape[0]={arr.shape[0]} != nTime={n_time}")

    for name, expected_fill in _PRIMITIVE_ARRAYS.items():
        if name not in names:
            violations.append(f"missing required array '{name}' (schema §3)")
            continue
        arr = root[name]
        if str(arr.dtype) != "uint16":
            violations.append(f"{name}: dtype={arr.dtype}, expected uint16 (schema §3 PRIMITIVE)")
        if arr.ndim != 2:
            violations.append(f"{name}: ndim={arr.ndim}, expected 2 (nTime, nNode)")
        elif n_time is not None and n_node is not None and arr.shape != (n_time, n_node):
            violations.append(f"{name}: shape={arr.shape}, expected ({n_time}, {n_node})")
        if arr.metadata.fill_value != expected_fill:
            violations.append(
                f"{name}: fill_value={arr.metadata.fill_value!r}, expected {expected_fill!r} "
                "(schema §1 — fill_value is load-bearing, must decode to physical zero)"
            )
        codecs_dumped = [c.to_dict() for c in arr.metadata.codecs]
        if codecs_dumped != _EXPECTED_CODECS:
            violations.append(f"{name}: codecs={codecs_dumped}, expected {_EXPECTED_CODECS} (schema §1)")
        cke = arr.metadata.chunk_key_encoding.to_dict()
        if cke != _EXPECTED_CHUNK_KEY_ENCODING:
            violations.append(f"{name}: chunk_key_encoding={cke}, expected {_EXPECTED_CHUNK_KEY_ENCODING}")
        if declared_chunk_length_t is not None:
            if arr.chunks[0] != declared_chunk_length_t:
                violations.append(
                    f"{name}: time-chunk length={arr.chunks[0]}, expected "
                    f"{declared_chunk_length_t} (the store's own declared "
                    "chunk_length_t, O1/D5)"
                )
        elif arr.chunks[0] != 10:
            violations.append(f"{name}: time-chunk length={arr.chunks[0]}, expected 10 (O1, v1 store)")

        for qattr in _REQUIRED_QUANT_ATTRS:
            if qattr not in arr.attrs:
                violations.append(f"{name}: missing quantization attr '{qattr}' (schema §3)")

        scale = arr.attrs.get("scale")
        valid_min = arr.attrs.get("valid_min")
        valid_max = arr.attrs.get("valid_max")
        if scale is not None and (not isinstance(scale, (int, float)) or scale == 0):
            violations.append(f"{name}: scale={scale!r} — degenerate-range guard violated (schema §3)")
        if valid_min is not None and valid_max is not None:
            import math

            if not (math.isfinite(valid_min) and math.isfinite(valid_max)):
                violations.append(f"{name}: valid_min/valid_max non-finite (schema §3 assert)")
            elif valid_max < valid_min:
                violations.append(f"{name}: valid_max < valid_min — range inverted (schema §3 assert)")

    if _INRADIUS_ARRAY not in names:
        violations.append(f"missing required array '{_INRADIUS_ARRAY}' (schema §2)")
    else:
        arr = root[_INRADIUS_ARRAY]
        if str(arr.dtype) != "float32":
            violations.append(f"{_INRADIUS_ARRAY}: dtype={arr.dtype}, expected float32 (schema §2)")
        if n_face is not None and arr.shape[0] != n_face:
            violations.append(
                f"{_INRADIUS_ARRAY}: shape[0]={arr.shape[0]} != nFace={n_face} "
                "(schema §2 — per-triangle, NOT per-node)"
            )

    return violations


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m run_anuga.validate_playback_store <store_path>", file=sys.stderr)
        return 2
    store_path = Path(argv[0])
    violations = validate_store(store_path)
    if violations:
        print(f"INVALID — {len(violations)} violation(s) against schema v1:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print(f"VALID — {store_path} conforms to schema v1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
