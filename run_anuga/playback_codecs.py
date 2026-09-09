"""The ``temporal_delta`` zarr v3 codec (TASK-2989, W3.1, epic 2981).

WHAT IT DOES. A playback time-series chunk is ``[chunk_length_t, n_node]``
uint16: the SAME nodes, one row per timestep. Consecutive rows are nearly
identical — water moves slowly relative to the output cadence — but gzip cannot
see that, because one row of a prod-scale mesh is 6.8 MB and DEFLATE's window is
32 KiB. Subtracting the previous row first turns "almost the same number" into
"a very small number", which is exactly what DEFLATE is good at.

    encode:  row 0 raw, row i -> row i - row (i-1), modulo 2**16
    decode:  cumulative sum along axis 0, modulo 2**16

LOSSLESS BY CONSTRUCTION, and that is why the arithmetic WRAPS instead of
clipping. Unsigned wraparound is exactly invertible; a clip to [0, 65535] would
throw away the sign of every falling value and silently render the wrong water.

WHY IT IS GATED AT CHUNK LENGTH >= 5 (decision D5). The gain comes from
inter-row similarity, and a short chunk has almost no rows to exploit while
still paying the delta's cost in entropy on row 0's successor. Measured on the
real stores:

    run 1412 chunk 5, length 10   depth      789,579 -> 277,225 B  (35%)
                                  x_velocity 590,499 -> 283,597 B  (48%)
    run 1328 chunk 1, length 10   depth      2.66 MiB -> 1.84 MiB  (69%)
    run 1328 chunk-2 fixture      depth      172,709 -> 166,530 B  (96%)
                                  x_velocity 173,538 -> 173,592 B  (LARGER)

So at the D5 floor it buys nothing and can lose. ``export_playback_store``
applies it only when ``derive_chunk_length_t(n_node) >= 5``; below that the
store is byte-identical to what a pre-TASK-2989 export wrote.

THE ENTRY POINT IS NOT OPTIONAL. ``pyproject.toml`` declares this class under
``[project.entry-points."zarr.codecs"]``, so ANY zarr reader in an environment
where run_anuga is installed resolves ``temporal_delta`` without importing this
module — the web box's editable install, the rig, ``prod_store_dequant_check``.
Without it, ``zarr.registry.get_codec_class`` raises ``KeyError`` and every
Python reader in the fleet crashes on a v3 store.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from zarr.abc.codec import ArrayArrayCodec
from zarr.core.common import JSON, parse_named_configuration
from zarr.registry import register_codec

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import NDBuffer
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType
    from zarr.core.metadata.v3 import ChunkGridMetadata

#: The codec's name in zarr v3 metadata, in the entry-point group, and in the
#: manifest's `codecs` block. One constant so the exporter, the validator, the
#: entry point and the client's guard cannot drift apart.
TEMPORAL_DELTA_NAME = "temporal_delta"

#: D5 — the shortest time-chunk the codec is allowed on. See the module
#: docstring's measured table: at length 2 it is 96% on depth and LARGER on
#: velocity, i.e. a regression on exactly the biggest stores.
TEMPORAL_DELTA_MIN_CHUNK_LENGTH = 5


def _check_unsigned(chunk: np.ndarray) -> None:
    if chunk.dtype.kind != "u":
        raise ValueError(
            f"temporal_delta requires an unsigned integer dtype (got {chunk.dtype}). "
            "The transform is only exactly invertible where wraparound is defined; on a "
            "float or signed array it would be lossy, which for playback means rendering "
            "the wrong water rather than failing."
        )


def encode_temporal_delta(chunk: np.ndarray) -> np.ndarray:
    """Row 0 raw, every later row minus its predecessor, modulo the dtype.

    Pure numpy so it is testable without zarr's buffer machinery, and so the
    measurement scripts and the codec can never disagree about the transform.
    """
    chunk = np.asarray(chunk)
    _check_unsigned(chunk)
    if chunk.ndim < 1 or chunk.shape[0] < 2:
        return chunk.copy()
    out = np.empty_like(chunk)
    out[0] = chunk[0]
    # Unsigned subtraction wraps in numpy, which is precisely the inverse of
    # the cumulative sum below. NOT np.diff: that would promote and could
    # produce negatives the dtype cannot hold.
    out[1:] = chunk[1:] - chunk[:-1]
    return out


def decode_temporal_delta(chunk: np.ndarray) -> np.ndarray:
    """Cumulative sum along axis 0, modulo the dtype — the exact inverse."""
    chunk = np.asarray(chunk)
    _check_unsigned(chunk)
    if chunk.ndim < 1 or chunk.shape[0] < 2:
        return chunk.copy()
    # `dtype=chunk.dtype` keeps the accumulator in the stored width, so it wraps
    # the same way the encoder's subtraction did. Accumulating in int64 and
    # casting back would give the same answer here but only by accident, and
    # would allocate 4x the chunk.
    return np.cumsum(chunk, axis=0, dtype=chunk.dtype)


@dataclass(frozen=True)
class TemporalDeltaCodec(ArrayArrayCodec):
    """Zarr v3 array->array codec: wrapped first differences along axis 0.

    Configuration-free on purpose. The axis is not a parameter because a
    playback time-series chunk's axis 0 is the time axis by the schema's own
    definition, and a codec that could be pointed at the NODE axis would be one
    typo away from a store that decodes to plausible, wrong water.
    """

    is_fixed_size = True

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        parse_named_configuration(data, TEMPORAL_DELTA_NAME, require_configuration=False)
        return cls()

    def to_dict(self) -> dict[str, JSON]:
        return {"name": TEMPORAL_DELTA_NAME}

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGridMetadata,
    ) -> None:
        native = dtype.to_native_dtype()
        if native.kind != "u":
            raise ValueError(
                f"{TEMPORAL_DELTA_NAME} only supports unsigned integer data types, got {native}."
            )
        if len(shape) < 2:
            raise ValueError(
                f"{TEMPORAL_DELTA_NAME} needs a time axis to difference along; got shape {shape}."
            )

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        # Shape, dtype and fill value are all unchanged — this rearranges
        # values within the chunk and nothing else.
        return chunk_spec

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length

    def _encode_sync(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer | None:
        arr = cast("np.ndarray[Any, np.dtype[Any]]", chunk_array.as_ndarray_like())
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(encode_temporal_delta(arr))

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        return self._encode_sync(chunk_array, chunk_spec)

    def _decode_sync(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        arr = cast("np.ndarray[Any, np.dtype[Any]]", chunk_array.as_ndarray_like())
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(decode_temporal_delta(arr))

    async def _decode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> NDBuffer:
        return self._decode_sync(chunk_array, chunk_spec)


# Registered on import as well as through the entry point. The entry point is
# what makes a reader that has never heard of run_anuga work; this line is what
# makes an in-process caller that imported the module work even in an
# environment where the distribution metadata is stale (an editable install
# whose entry points were declared after `pip install -e`, which is exactly the
# state this box was in before TASK-2989).
register_codec(TEMPORAL_DELTA_NAME, TemporalDeltaCodec)
