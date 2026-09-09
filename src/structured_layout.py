"""Structure-aware, reversible layouts for XnYZip particle packages.

The target snapshots carry Lagrangian mesh coordinates in their particle IDs
while XnYZip emits particles in Eulerian spatial order.  This module keeps
both structures explicit without storing a permutation sidecar:

* IDs are mapped to a 3-D Hilbert code before lossless compression.
* Velocities are grouped by a coarse decoded-position cell and ordered within
  each cell by the particle's Lagrangian Morton code.

Every transform is integer-reversible or recomputable from decoded positions
and IDs, so it does not relax the pipeline's existing error guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from src.constants import POSITION_FIELDS
from src.lattice_layout import (
    LatticeLayoutUnavailable,
    decode_lattice_coordinates,
)


LATTICE_HILBERT_ID_CODEC = "lattice_hilbert_pcodec_v1"
HYBRID_VELOCITY_LAYOUT = "eulerian_lagrangian_hybrid_v1"
MAX_FAST_HILBERT_BITS = 10


@dataclass(frozen=True)
class StructuredParticleLayout:
    """Geometry needed by the structure-aware ID and velocity transforms."""

    side: int
    id_base: int
    position_digit_axes: Tuple[int, int, int]
    lattice_bits: int
    velocity_cell_bits: int

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": HYBRID_VELOCITY_LAYOUT,
            "enabled": True,
            "side": self.side,
            "id_base": self.id_base,
            "position_digit_axes": list(self.position_digit_axes),
            "lattice_bits": self.lattice_bits,
            "velocity_cell_bits": self.velocity_cell_bits,
            "id_transform": LATTICE_HILBERT_ID_CODEC,
            "velocity_order": (
                "coarse_eulerian_morton_then_lagrangian_morton"
            ),
            "permutation_sidecar": False,
            "position_source": "decoded_xnyzip_float32",
        }

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any],
    ) -> "StructuredParticleLayout":
        if not metadata.get("enabled", False):
            raise RuntimeError(
                "Manifest does not describe an enabled structure-aware layout."
            )
        if metadata.get("name") != HYBRID_VELOCITY_LAYOUT:
            raise RuntimeError("Unknown structure-aware layout version.")
        side = int(metadata["side"])
        id_base = int(metadata["id_base"])
        axes = tuple(int(value) for value in metadata["position_digit_axes"])
        lattice_bits = int(metadata["lattice_bits"])
        cell_bits = int(metadata["velocity_cell_bits"])
        expected_bits = lattice_code_bits(side)
        if id_base not in (0, 1):
            raise RuntimeError(f"Invalid structured ID base: {id_base}.")
        if sorted(axes) != [0, 1, 2]:
            raise RuntimeError(
                "Structured position_digit_axes must be a permutation of "
                "(0, 1, 2)."
            )
        if lattice_bits != expected_bits:
            raise RuntimeError(
                "Structured lattice_bits does not match the lattice side."
            )
        _validate_cell_bits(cell_bits, lattice_bits)
        return cls(side, id_base, axes, lattice_bits, cell_bits)


def make_structured_layout(
    side: int,
    id_base: int,
    position_digit_axes: Sequence[int],
    velocity_cell_bits: int,
) -> StructuredParticleLayout:
    """Validate and construct structure-aware layout metadata."""

    axes = tuple(int(value) for value in position_digit_axes)
    if id_base not in (0, 1):
        raise LatticeLayoutUnavailable(
            f"structured ID base must be zero or one, got {id_base}"
        )
    if sorted(axes) != [0, 1, 2]:
        raise LatticeLayoutUnavailable(
            "position-to-ID digit axes are not a permutation"
        )
    bits = lattice_code_bits(side)
    _validate_cell_bits(velocity_cell_bits, bits)
    return StructuredParticleLayout(
        int(side),
        int(id_base),
        axes,  # type: ignore[arg-type]
        bits,
        int(velocity_cell_bits),
    )


def validate_structured_package(manifest: Mapping[str, Any]) -> None:
    """Reject incomplete or contradictory metadata before decoding any fields."""

    metadata = manifest.get("structured_layout", {})
    fields = manifest.get("compressed_fields", {})
    hybrid_fields = [key for key in ("vx", "vy", "vz")
                     if fields.get(key, {}).get("spatial_layout") == HYBRID_VELOCITY_LAYOUT]
    hybrid_triplet = fields.get("velocities", {}).get("spatial_layout") == HYBRID_VELOCITY_LAYOUT
    structured_ids = fields.get("id", {}).get("codec") == LATTICE_HILBERT_ID_CODEC
    if not metadata.get("enabled", False):
        if hybrid_fields or hybrid_triplet or structured_ids:
            raise RuntimeError("Structured fields are missing enabled package layout metadata.")
        return
    layout = StructuredParticleLayout.from_metadata(metadata)
    szo_velocities = len(hybrid_fields) == 3 and all(
        fields[key].get("codec") == "szo" for key in hybrid_fields
    )
    xnyzip_velocities = (
        hybrid_triplet and fields["velocities"].get("codec") == "xnyzip"
        and "velocity_order" in fields
    )
    if not structured_ids or not (szo_velocities or xnyzip_velocities):
        raise RuntimeError("Structured package requires transformed IDs and three hybrid velocities.")
    id_layout = StructuredParticleLayout.from_metadata(fields["id"]["structured_layout"])
    if id_layout != layout:
        raise RuntimeError("Structured ID and velocity layout metadata disagree.")
    if fields.get("positions", {}).get("codec") != "xnyzip" or (hybrid_fields and hybrid_triplet):
        raise RuntimeError("Structured package requires XnYZip positions and SZO or XnYZip velocities.")


def lattice_code_bits(side: int) -> int:
    """Return bits per lattice axis supported by the fast Hilbert mapping."""

    side = int(side)
    if side <= 1:
        raise LatticeLayoutUnavailable(
            "structure-aware layout requires a lattice side greater than one"
        )
    bits = (side - 1).bit_length()
    if bits > MAX_FAST_HILBERT_BITS:
        raise LatticeLayoutUnavailable(
            "structure-aware layout currently supports lattice sides up to "
            f"{1 << MAX_FAST_HILBERT_BITS}"
        )
    return bits


def encode_lattice_ids(
    ids: np.ndarray,
    layout: StructuredParticleLayout,
) -> np.ndarray:
    """Map source IDs to physical-axis 3-D Hilbert codes."""

    coordinates = decode_lattice_coordinates(
        np.asarray(ids),
        layout.side,
        layout.id_base,
    )
    physical = tuple(
        coordinates[layout.position_digit_axes[axis]]
        for axis in range(3)
    )
    morton = morton_encode_3d(*physical)
    return morton_to_hilbert_3d(morton, layout.lattice_bits)


def decode_lattice_ids(
    codes: np.ndarray,
    layout: StructuredParticleLayout,
    output_dtype: np.dtype,
) -> np.ndarray:
    """Invert physical-axis Hilbert codes back to source particle IDs."""

    values = np.asarray(codes)
    if values.ndim != 1 or values.size == 0:
        raise RuntimeError("Structured ID codes are empty or not one-dimensional.")
    if values.dtype != np.dtype("uint32"):
        raise RuntimeError(
            f"Structured ID codes must use uint32, got {values.dtype}."
        )
    if int(values.max(initial=0)) >= (1 << (3 * layout.lattice_bits)):
        raise RuntimeError("Structured ID code exceeds the configured Hilbert capacity.")
    morton = hilbert_to_morton_3d(values, layout.lattice_bits)
    physical = morton_decode_3d(morton)
    if any(int(axis.max(initial=0)) >= layout.side for axis in physical):
        raise RuntimeError(
            "Structured ID payload decodes outside the configured lattice."
        )
    digits = [None, None, None]
    for position_axis, digit_axis in enumerate(layout.position_digit_axes):
        digits[digit_axis] = physical[position_axis].astype(
            np.uint64,
            copy=False,
        )
    high, middle, low = digits
    assert high is not None and middle is not None and low is not None
    linear = (
        (high * np.uint64(layout.side) + middle)
        * np.uint64(layout.side)
        + low
        + np.uint64(layout.id_base)
    )
    dtype = np.dtype(output_dtype)
    if not np.issubdtype(dtype, np.integer):
        raise RuntimeError(f"Structured IDs require an integer dtype, got {dtype}.")
    limits = np.iinfo(dtype)
    if int(linear.max(initial=0)) > int(limits.max):
        raise RuntimeError(
            f"Structured IDs do not fit reconstructed dtype {dtype}."
        )
    return linear.astype(dtype, copy=False)


def hybrid_velocity_order(
    ids: np.ndarray,
    decoded_positions: Mapping[str, np.ndarray],
    layout: StructuredParticleLayout,
) -> np.ndarray:
    """Return hybrid-index to canonical-index velocity permutation."""

    ids = np.asarray(ids)
    count = int(ids.size)
    if ids.ndim != 1 or count == 0:
        raise RuntimeError(
            "Hybrid velocity ordering requires a non-empty 1-D ID field."
        )
    coordinates = decode_lattice_coordinates(
        ids,
        layout.side,
        layout.id_base,
    )
    physical = tuple(
        coordinates[layout.position_digit_axes[axis]]
        for axis in range(3)
    )
    lagrangian = morton_encode_3d(*physical).astype(np.uint64, copy=False)

    cell_scale = 1 << layout.velocity_cell_bits
    cell_coordinates = []
    for logical in POSITION_FIELDS:
        if logical not in decoded_positions:
            raise RuntimeError(
                f"Hybrid velocity ordering is missing decoded position {logical}."
            )
        values = np.asarray(decoded_positions[logical])
        if values.ndim != 1 or values.size != count:
            raise RuntimeError(
                f"Decoded position {logical} has shape {values.shape}; "
                f"expected ({count},)."
            )
        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"Decoded position {logical} contains non-finite values."
            )
        wrapped = np.remainder(values.astype(np.float64), 1.0)
        cells = np.floor(wrapped * cell_scale).astype(np.uint32)
        np.minimum(cells, cell_scale - 1, out=cells)
        cell_coordinates.append(cells)
    eulerian = morton_encode_3d(*cell_coordinates).astype(
        np.uint64,
        copy=False,
    )
    key = (eulerian << np.uint64(3 * layout.lattice_bits)) | lagrangian
    return np.argsort(key, kind="stable").astype(np.intp, copy=False)


def morton_encode_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Interleave three at-most-10-bit coordinate arrays into uint32."""

    coordinates = tuple(np.asarray(axis) for axis in (x, y, z))
    for axis in coordinates:
        if axis.dtype.kind not in "iu" or np.any(axis < 0) or np.any(axis >= 1024):
            raise RuntimeError("Morton coordinates must be integers in [0, 1024).")
    x_values, y_values, z_values = (
        axis.astype(np.uint32, copy=False) for axis in coordinates)
    if not (
        x_values.shape == y_values.shape == z_values.shape
        and x_values.ndim == 1
    ):
        raise RuntimeError("Morton coordinates must be equally sized 1-D arrays.")
    return (
        _split_by_3(x_values)
        | (_split_by_3(y_values) << np.uint64(1))
        | (_split_by_3(z_values) << np.uint64(2))
    ).astype(np.uint32)


def morton_decode_3d(
    morton: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deinterleave uint32 Morton codes into three coordinate arrays."""

    values = np.asarray(morton, dtype=np.uint32)
    return (
        _compact_by_3(values),
        _compact_by_3(values >> np.uint32(1)),
        _compact_by_3(values >> np.uint32(2)),
    )


def morton_to_hilbert_3d(morton: np.ndarray, bits: int) -> np.ndarray:
    """Vectorized form of XnYZip's fastMortonToHilbert3_32."""

    _validate_hilbert_bits(bits)
    hilbert = np.asarray(morton, dtype=np.uint32).copy()
    if bits > 1:
        block = bits * 3 - 3
        hcode = (hilbert >> np.uint32(block)) & np.uint32(7)
        shift = np.zeros_like(hilbert)
        signs = np.zeros_like(hilbert)
        while block:
            block -= 3
            hcode <<= np.uint32(2)
            mcode = (np.uint32(0x20212021) >> hcode) & np.uint32(3)
            shift = (
                np.uint32(0x48)
                >> (np.uint32(7) - shift - mcode)
            ) & np.uint32(3)
            signs = (signs | (signs << np.uint32(3))) >> mcode
            signs = (
                signs ^ (np.uint32(0x53560300) >> hcode)
            ) & np.uint32(7)
            mcode = (hilbert >> np.uint32(block)) & np.uint32(7)
            hcode = mcode.copy()
            hcode = (
                (hcode | (hcode << np.uint32(3))) >> shift
            ) & np.uint32(7)
            hcode ^= signs
            hilbert ^= (mcode ^ hcode) << np.uint32(block)
    hilbert ^= (hilbert >> np.uint32(1)) & np.uint32(0x92492492)
    hilbert ^= (hilbert & np.uint32(0x92492492)) >> np.uint32(1)
    return hilbert


def hilbert_to_morton_3d(hilbert: np.ndarray, bits: int) -> np.ndarray:
    """Vectorized form of XnYZip's fastHilbertToMorton3_32."""

    _validate_hilbert_bits(bits)
    morton = np.asarray(hilbert, dtype=np.uint32).copy()
    morton ^= (morton & np.uint32(0x92492492)) >> np.uint32(1)
    morton ^= (morton >> np.uint32(1)) & np.uint32(0x92492492)
    if bits > 1:
        block = bits * 3 - 3
        hcode = (morton >> np.uint32(block)) & np.uint32(7)
        shift = np.zeros_like(morton)
        signs = np.zeros_like(morton)
        while block:
            block -= 3
            hcode <<= np.uint32(2)
            mcode = (np.uint32(0x20212021) >> hcode) & np.uint32(3)
            shift = (
                np.uint32(0x48)
                >> (np.uint32(4) - shift + mcode)
            ) & np.uint32(3)
            signs = (signs | (signs << np.uint32(3))) >> mcode
            signs = (
                signs ^ (np.uint32(0x53560300) >> hcode)
            ) & np.uint32(7)
            hcode = (morton >> np.uint32(block)) & np.uint32(7)
            mcode = hcode ^ signs
            mcode = (
                (mcode | (mcode << np.uint32(3))) >> shift
            ) & np.uint32(7)
            morton ^= (hcode ^ mcode) << np.uint32(block)
    return morton


def _split_by_3(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.uint64, copy=True) & np.uint64(0x1FFFFF)
    result = (result | (result << np.uint64(32))) & np.uint64(
        0x1F00000000FFFF
    )
    result = (result | (result << np.uint64(16))) & np.uint64(
        0x1F0000FF0000FF
    )
    result = (result | (result << np.uint64(8))) & np.uint64(
        0x100F00F00F00F00F
    )
    result = (result | (result << np.uint64(4))) & np.uint64(
        0x10C30C30C30C3
    )
    return (result | (result << np.uint64(2))) & np.uint64(
        0x1249249249249249
    )


def _compact_by_3(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.uint64, copy=True) & np.uint64(
        0x1249249249249249
    )
    result = (result | (result >> np.uint64(2))) & np.uint64(
        0x10C30C30C30C3
    )
    result = (result | (result >> np.uint64(4))) & np.uint64(
        0x100F00F00F00F00F
    )
    result = (result | (result >> np.uint64(8))) & np.uint64(
        0x1F0000FF0000FF
    )
    result = (result | (result >> np.uint64(16))) & np.uint64(
        0x1F00000000FFFF
    )
    result = (result | (result >> np.uint64(32))) & np.uint64(0x1FFFFF)
    return result.astype(np.uint32)


def _validate_hilbert_bits(bits: int) -> None:
    if not 1 <= int(bits) <= MAX_FAST_HILBERT_BITS:
        raise RuntimeError(
            f"Hilbert bits must be in [1, {MAX_FAST_HILBERT_BITS}], got {bits}."
        )


def _validate_cell_bits(cell_bits: int, lattice_bits: int) -> None:
    cell_bits = int(cell_bits)
    if not 1 <= cell_bits <= MAX_FAST_HILBERT_BITS:
        raise LatticeLayoutUnavailable(
            "structure-aware velocity cell bits must be in "
            f"[1, {MAX_FAST_HILBERT_BITS}]"
        )
    if 3 * (cell_bits + lattice_bits) > 64:
        raise LatticeLayoutUnavailable(
            "structured velocity composite key exceeds 64 bits"
        )


__all__ = [
    "HYBRID_VELOCITY_LAYOUT",
    "LATTICE_HILBERT_ID_CODEC",
    "StructuredParticleLayout",
    "decode_lattice_ids",
    "encode_lattice_ids",
    "hilbert_to_morton_3d",
    "hybrid_velocity_order",
    "lattice_code_bits",
    "make_structured_layout",
    "morton_decode_3d",
    "morton_encode_3d",
    "morton_to_hilbert_3d",
]
