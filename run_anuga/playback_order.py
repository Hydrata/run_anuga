"""Spatial (Morton / Z-order) node ordering for the playback store.

TASK-3014 (W3.0, epic 2981), Option A of
``docs/reports/2026-09-08-epic-2981-device-adaptation-critique.html``.

WHY THIS EXISTS. On the big prod store, geometry is 60 of 83 MiB (72%) of the
bytes and it is a BLOCKING PREFIX — nothing plays until it has all arrived
(101 s at 5 Mbit/s, four minutes at 2 Mbit/s). The SWW writes nodes in mesh-
generation order, which is spatially incoherent, so ``face_node_connectivity``
holds three large, uncorrelated integers per triangle and ``elevation`` holds a
sequence with no local structure. DEFLATE can only exploit similarity inside
its 32 KiB window, so it finds almost none. Reordering the nodes along a
Z-curve puts spatial neighbours next to each other in the array, which is what
gives the window something to match.

CLIENT-TRANSPARENT BY CONSTRUCTION. The browser reads only the store's own
arrays: node_x/node_y/elevation/friction, the node axis of the time series,
and face_node_connectivity's VALUES. Permute every one of those consistently
and the rendered mesh is identical — no manifest change, no decode change, no
format-version bump. That is the whole reason this lever is separate from the
codec-chain work (W3.1/W3.2/W3.3), which is NOT transparent.

THE PERMUTATION CONVENTION, stated once because both directions are needed and
mixing them silently corrupts geometry:

    perm[newIndex] = originalIndex          (what morton_node_order returns,
                                             and what the store writes as
                                             ``node_permutation``)
    inv[originalIndex] = newIndex           (inverse_permutation(perm))

So a per-node array is reordered with ``a[perm]``, and an array that CONTAINS
node indices — ``face_node_connectivity`` — is remapped with ``inv[values]``.
Getting these the wrong way round produces a store that still validates and
still renders, just with the mesh scrambled.
"""
from __future__ import annotations

import numpy as np

#: Bits per axis in the quantized lattice the Z-curve is computed on. 16 gives
#: a 65,536 x 65,536 grid over the store's own bounding box — finer than any
#: mesh we export (run 1328 is 3.4 M nodes, i.e. ~1,842 nodes per axis if it
#: were a square lattice), so ties are rare and, where they happen, broken
#: deterministically by the stable argsort below.
MORTON_BITS = 16
#: The declared value of the ``node_order`` / ``face_order`` group attrs.
#: ABSENT means "the original SWW order" — that is the first-class-absence
#: shape ``has_dt`` and ``envelope_quantities`` already use, and it is what
#: keeps every store written before this task valid forever.
MORTON_ORDER_NAME = "morton"


def _quantize_axis(values: np.ndarray, bits: int) -> np.ndarray:
    """Map ``values`` onto ``[0, 2**bits - 1]`` over their own min/max span.

    A DEGENERATE span (every node on one coordinate — a legal, if useless,
    mesh, and exactly the shape ``quantize_range``'s guard exists for) divides
    by zero and would produce NaN, which ``argsort`` orders arbitrarily and
    which would therefore break determinism. Answer zeros instead: the axis
    then contributes nothing to the key and the other axis decides.
    """
    grid_max = (1 << bits) - 1
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=np.uint64)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    span = hi - lo
    if not np.isfinite(span) or span <= 0.0:
        return np.zeros(values.shape[0], dtype=np.uint64)
    scaled = np.rint((values - lo) / span * grid_max)
    # NaN coordinates cannot be placed on the curve; park them at the origin
    # rather than letting them poison the sort order.
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=grid_max, neginf=0.0)
    return np.clip(scaled, 0, grid_max).astype(np.uint64)


def _part1by1(value: np.ndarray) -> np.ndarray:
    """Spread the low 16 bits of each element out into every other bit slot
    (the standard "magic bits" interleave), so ``_part1by1(x) |
    (_part1by1(y) << 1)`` is the 32-bit Morton code of ``(x, y)``.
    """
    value = value & np.uint64(0x0000FFFF)
    value = (value | (value << np.uint64(8))) & np.uint64(0x00FF00FF)
    value = (value | (value << np.uint64(4))) & np.uint64(0x0F0F0F0F)
    value = (value | (value << np.uint64(2))) & np.uint64(0x33333333)
    value = (value | (value << np.uint64(1))) & np.uint64(0x55555555)
    return value


def _morton_keys(x, y, bits: int) -> np.ndarray:
    """The Z-order key of every ``(x, y)``, on a ``2**bits`` lattice over the
    points' own bounding box.

    Private: :func:`morton_node_order` is the whole public surface, and an
    exported key function invites a second caller to sort them itself with a
    non-stable sort — which would break the reproducibility the stored
    ``node_permutation`` depends on.
    """
    xi = _quantize_axis(x, bits)
    yi = _quantize_axis(y, bits)
    return _part1by1(xi) | (_part1by1(yi) << np.uint64(1))


def morton_node_order(x, y, bits: int = MORTON_BITS) -> np.ndarray:
    """``perm`` such that ``perm[newIndex] == originalIndex``.

    DETERMINISTIC AND STABLE — the same inputs give the same permutation on
    every run and on every box, so a re-export is reproducible and the stored
    ``node_permutation`` can be trusted as the one way back to SWW order.
    ``kind='stable'`` is load-bearing for that: quantization ties are broken by
    original index, never by whatever order the sort happened to visit.

    @param x per-node coordinate array (any float dtype)
    @param y per-node coordinate array, same length
    @returns int64 array of length ``len(x)``
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError(f"morton_node_order: x{x.shape} and y{y.shape} must have the same shape")
    if x.ndim != 1:
        raise ValueError(f"morton_node_order: expected 1-D coordinates, got ndim={x.ndim}")
    if x.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.argsort(_morton_keys(x, y, bits), kind="stable").astype(np.int64)


def inverse_permutation(perm) -> np.ndarray:
    """``inv`` such that ``inv[originalIndex] == newIndex``.

    This is the array ``face_node_connectivity``'s VALUES are remapped
    through: a triangle holds original node indices, and after the reorder it
    must hold the new ones. Triangle WINDING is untouched by construction —
    this is a value substitution, never a sort of the three vertices within a
    row, and the renderer's orientation depends on that.
    """
    perm = np.asarray(perm)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.shape[0], dtype=perm.dtype)
    return inv


def is_permutation(values, n: int) -> bool:
    """True iff ``values`` is a genuine permutation of ``0..n-1``.

    The validator's check (TASK-3014 AC5). The failure mode this catches is a
    DUPLICATED index — a permutation with a repeat still has the right dtype,
    the right shape and an entirely plausible value range, but it drops one
    node and doubles another, so a re-export through it silently corrupts the
    mesh. Nothing else in the store would notice.
    """
    values = np.asarray(values)
    if values.ndim != 1 or values.shape[0] != n:
        return False
    if values.dtype.kind not in ("i", "u"):
        return False
    return bool(np.array_equal(np.sort(values), np.arange(n, dtype=values.dtype)))
