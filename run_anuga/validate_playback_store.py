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

# TASK-2752 (W8.2, epic 2706) — imported (not re-declared) so the producer's
# and the validator's idea of "which quantities can have an envelope" can
# never drift apart. Safe: playback_store.py's only module-level import is
# numpy, and it never imports this module (no cycle).
from run_anuga.playback_store import ENVELOPE_QUANTITIES

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
#: TASK-3014 (W3.0, epic 2981) — the spatial-order declarations, as
#: ``(group attr, permutation array, "which count it must permute")`` triples.
#: PRESENCE-GATED, exactly like ``has_dt`` and ``envelope_quantities``: a store
#: that declares NEITHER is in the original SWW order and must validate with
#: ZERO violations forever. Declaring the attr is what makes the array
#: required, and vice versa — a half-declared store is refused in both
#: directions, because either half alone silently misdescribes the mesh.
_ORDER_DECLARATIONS = (
    ("node_order", "node_permutation", "nNode"),
    ("face_order", "face_permutation", "nFace"),
)
#: The only ordering this validator knows how to reason about. An unknown name
#: is refused rather than ignored: "hilbert" would mean the stored permutation
#: does not describe the curve the attr claims, and a reader that trusted the
#: attr would re-export a scrambled mesh.
_KNOWN_ORDERS = ("morton",)


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
        isinstance(declared_chunk_length_t, int)
        and _CHUNK_LENGTH_T_FLOOR <= declared_chunk_length_t <= _CHUNK_LENGTH_T_CAP
    ):
        violations.append(
            f"chunk_length_t={declared_chunk_length_t!r}, expected an int in "
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

    # TASK-2752 (v2, epic 2706 W8.2) — envelope arrays are OPTIONAL, declared
    # capabilities (the has_dt shape): a v2 store that declares NONE is fully
    # valid (zero violations from this block). Only a DECLARED-but-missing
    # (or malformed) envelope is a violation — "declares an envelope it does
    # not contain" is refused, absence itself is not.
    declared_envelopes = root.attrs.get("envelope_quantities")
    if declared_envelopes:
        if not isinstance(declared_envelopes, (list, tuple)):
            violations.append(
                f"envelope_quantities={declared_envelopes!r}, expected a list (TASK-2752, schema §5)"
            )
        else:
            for env_name in declared_envelopes:
                if env_name not in ENVELOPE_QUANTITIES:
                    violations.append(
                        f"envelope_quantities declares unknown quantity '{env_name}', expected one of "
                        f"{ENVELOPE_QUANTITIES} (TASK-2752)"
                    )
                    continue
                arr_name = f"{env_name}_max"
                if arr_name not in names:
                    violations.append(
                        f"group attrs declare envelope '{env_name}' but array '{arr_name}' is missing "
                        "(TASK-2752 — a store must not advertise a capability it does not have)"
                    )
                    continue
                arr = root[arr_name]
                if str(arr.dtype) != "uint16":
                    violations.append(f"{arr_name}: dtype={arr.dtype}, expected uint16 (TASK-2752)")
                if arr.ndim != 1:
                    violations.append(f"{arr_name}: ndim={arr.ndim}, expected 1 — (nNode,) (TASK-2752)")
                elif n_node is not None and arr.shape[0] != n_node:
                    violations.append(
                        f"{arr_name}: shape[0]={arr.shape[0]} != nNode={n_node} (TASK-2752)"
                    )
                for qattr in _REQUIRED_QUANT_ATTRS:
                    if qattr not in arr.attrs:
                        violations.append(
                            f"{arr_name}: missing quantization attr '{qattr}' (TASK-2752, schema §3)"
                        )

    # TASK-3014 (W3.0, epic 2981) — spatial (Morton) export order.
    counts = {"nNode": n_node, "nFace": n_face}
    for attr_name, array_name, count_key in _ORDER_DECLARATIONS:
        declared_order = root.attrs.get(attr_name)
        present = array_name in names
        if declared_order is None:
            if present:
                violations.append(
                    f"array '{array_name}' is present but group attrs declare no "
                    f"'{attr_name}' — a permutation nothing declares cannot be applied, "
                    "and absence of the attr means the ORIGINAL SWW order (TASK-3014)"
                )
            continue
        if declared_order not in _KNOWN_ORDERS:
            violations.append(
                f"{attr_name}={declared_order!r}, expected one of {_KNOWN_ORDERS} "
                "(TASK-3014 — an ordering this reader cannot reproduce)"
            )
        if not present:
            violations.append(
                f"group attrs declare {attr_name}={declared_order!r} but array "
                f"'{array_name}' is missing (TASK-3014 — a store must not advertise "
                "a reordering it cannot undo)"
            )
            continue
        arr = root[array_name]
        if str(arr.dtype) != "int32":
            violations.append(
                f"{array_name}: dtype={arr.dtype}, expected int32 (TASK-3014)"
            )
        expected_count = counts.get(count_key)
        if expected_count is None:
            continue
        if arr.ndim != 1 or arr.shape[0] != expected_count:
            violations.append(
                f"{array_name}: shape={arr.shape}, expected ({expected_count},) "
                f"— one entry per {count_key} (TASK-3014)"
            )
            continue
        # THE check that matters. A DUPLICATED index keeps the dtype, the
        # shape and an entirely plausible value range, but drops one node and
        # doubles another — a re-export through it silently corrupts the mesh
        # and nothing else in the store would notice.
        from run_anuga.playback_order import is_permutation

        if not is_permutation(arr[:], expected_count):
            violations.append(
                f"{array_name}: not a permutation of 0..{expected_count - 1} "
                "(duplicated or out-of-range index) — re-exporting through it would "
                "silently corrupt the mesh (TASK-3014)"
            )

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
