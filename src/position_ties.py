"""Encoder-only velocity ordering inside identical decoded position runs.

Every row in a run has the same *decoded* position bits. Permuting whole
particle records inside that run therefore leaves the position stream intact,
including its error guarantee. Only the temporary source-row mapping changes;
there is no new package stream or decoder operation.
"""

from typing import Mapping

import numpy as np

from src.constants import POSITION_FIELDS, VELOCITY_FIELDS
from src.structured_layout import morton_encode_3d, morton_to_hilbert_3d


TIE_SORT_BLOCK_SIZE = 1_048_576
TIE_SORT_BITS = 10


def sort_position_ties(
    order: np.ndarray,
    decoded_positions: Mapping[str, np.ndarray],
    source_velocities: Mapping[str, np.ndarray],
    *,
    block_size: int = TIE_SORT_BLOCK_SIZE,
) -> dict:
    """Refine a validated native permutation in place, using bounded scratch.

    Positions are in native decoded order; velocities are in original source
    order. Fixed blocks may split a run, which only limits compression gain.
    Ties in the Hilbert key retain native order. Normalization affects sorting
    alone: no velocity values are quantized or changed here.
    """
    count = len(order)
    if block_size < 1:
        raise ValueError("Position tie block size must be positive.")
    for key in POSITION_FIELDS:
        values = decoded_positions[key]
        if values.shape != (count,) or values.dtype != np.dtype("float32"):
            raise ValueError("Decoded positions must be float32 vectors matching the order.")
    for key in VELOCITY_FIELDS:
        if source_velocities[key].shape != (count,):
            raise ValueError("Source velocities must match the order length.")

    moved = 0
    tied = 0
    for start in range(0, count, block_size):
        end = min(count, start + block_size)
        length = end - start
        change = np.zeros(length - 1, dtype=bool)
        for key in POSITION_FIELDS:
            bits = decoded_positions[key][start:end].view(np.uint32)
            change |= bits[1:] != bits[:-1]
        tied += int(np.count_nonzero(~change))
        if change.all():
            continue
        groups = np.empty(length, dtype=np.uint32)
        groups[0] = 0
        np.cumsum(change, dtype=np.uint32, out=groups[1:])
        source_rows = order[start:end].copy()
        coordinates = []
        for key in VELOCITY_FIELDS:
            values = source_velocities[key][source_rows].astype(np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError("Position tie sorting requires finite velocities.")
            lower, upper = float(values.min()), float(values.max())
            if upper == lower:
                coordinates.append(np.zeros(length, dtype=np.uint32))
            else:
                normalized = (values - lower) / (upper - lower)
                coordinates.append(
                    np.floor(normalized * ((1 << TIE_SORT_BITS) - 1)).astype(np.uint32)
                )
        key = morton_to_hilbert_3d(morton_encode_3d(*coordinates), TIE_SORT_BITS)
        permutation = np.lexsort((key, groups))
        # An executable invariant guards the particle-association contract.
        if not np.array_equal(groups[permutation], groups):
            raise RuntimeError("Position tie ordering crossed a decoded position run.")
        moved += int(np.count_nonzero(permutation != np.arange(length)))
        order[start:end] = source_rows[permutation]

    return {
        "enabled": True,
        "method": "decoded_position_ties_velocity_hilbert",
        "hilbert_bits": TIE_SORT_BITS,
        "block_size": block_size,
        "equal_adjacent_pairs_in_blocks": tied,
        "moved_particles": moved,
        "sidecar_bytes": 0,
    }
