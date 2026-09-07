"""Periodic ID-lattice transforms for fieldwise particle compression.

The transform is package-local and reversible from the losslessly stored,
ID-sorted particle IDs.  It unwraps each periodic lattice axis at its largest
empty gap, scatters occupied values into the resulting dense 3-D box, and
linearly fills holes solely to improve prediction.  Missing-cell values are
never exposed by decompression.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.constants import MIN_CODEC_VALUES, POSITION_FIELDS


LATTICE_LAYOUT_NAME = "periodic_dense_id_lattice_v1"
POSITION_RESIDUAL_TRANSFORM = "periodic_lattice_residual_v1"
IDENTITY_TRANSFORM = "identity"
IMPLICIT_CHUNK_ROWS = 4_194_304


class LatticeLayoutUnavailable(RuntimeError):
    """Raised when a source cannot safely use the dense lattice transform."""


@dataclass(frozen=True)
class DenseLatticeLayout:
    """Resolved lattice geometry plus transient gather/scatter indices."""

    side: int
    id_base: int
    starts: Tuple[int, int, int]
    shape: Tuple[int, int, int]
    position_digit_axes: Tuple[int, int, int]
    count: int
    dense_indices: np.ndarray
    coordinates: Tuple[np.ndarray, np.ndarray, np.ndarray]
    dense_order: np.ndarray
    missing_indices: np.ndarray
    implicit_full_lattice: bool = False

    @property
    def dense_count(self) -> int:
        return math.prod(self.shape)

    @property
    def occupancy(self) -> float:
        return self.count / self.dense_count

    def manifest_metadata(self) -> Dict[str, Any]:
        return {
            "name": LATTICE_LAYOUT_NAME,
            "enabled": True,
            "side": self.side,
            "id_base": self.id_base,
            "periodic_starts": list(self.starts),
            "dense_shape": list(self.shape),
            "dense_count": self.dense_count,
            "particle_count": self.count,
            "occupancy": self.occupancy,
            "position_digit_axes": list(self.position_digit_axes),
            "hole_fill": "linear_flat_index",
            "implicit_full_lattice": self.implicit_full_lattice,
        }

    def encode_field(
        self,
        values: np.ndarray,
        logical: str,
        position_residual: bool,
    ) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
        """Return a dense base-axis array and no-codec transform error."""

        values = np.asarray(values)
        if values.ndim != 1 or values.size != self.count:
            raise RuntimeError(
                f"Lattice field {logical} expected {self.count} values, "
                f"got shape {values.shape}."
            )
        if self.implicit_full_lattice:
            return self._encode_implicit_full_field(
                values,
                logical,
                position_residual,
            )
        transformed = values
        roundtrip_error = 0.0
        wrap_offsets: Optional[np.ndarray] = None
        if position_residual:
            try:
                position_axis = POSITION_FIELDS.index(logical)
            except ValueError as exc:
                raise RuntimeError(
                    f"Lattice residual requested for non-position field {logical}."
                ) from exc
            digit_axis = self.position_digit_axes[position_axis]
            predictor = (
                self.coordinates[digit_axis].astype(np.float64) / self.side
            )
            unwrapped_residual = values.astype(np.float64) - predictor
            wrapped_residual = np.remainder(
                unwrapped_residual + 0.5,
                1.0,
            ) - 0.5
            wrap_offsets = np.rint(
                unwrapped_residual - wrapped_residual
            ).astype(np.int8)
            if np.any(np.abs(wrap_offsets) > 1):
                raise RuntimeError(
                    f"Unexpected periodic wrap offset for lattice field {logical}."
                )
            transformed = wrapped_residual.astype(np.float32)
            restored = (
                transformed.astype(np.float64)
                + predictor
                + wrap_offsets.astype(np.float64)
            ).astype(np.float32)
            roundtrip_error = float(
                np.max(
                    np.abs(
                        restored.astype(np.float64)
                        - values.astype(np.float64)
                    ),
                    initial=0.0,
                )
            )

        dense = np.empty(self.dense_count, dtype=transformed.dtype)
        dense[self.dense_indices] = transformed
        if self.missing_indices.size:
            sorted_indices = self.dense_indices[self.dense_order]
            sorted_values = transformed[self.dense_order]
            dense[self.missing_indices] = np.interp(
                self.missing_indices,
                sorted_indices,
                sorted_values,
            ).astype(transformed.dtype)
        return dense.reshape(self.shape), roundtrip_error, wrap_offsets

    def _encode_implicit_full_field(
        self,
        values: np.ndarray,
        logical: str,
        position_residual: bool,
    ) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
        if not position_residual:
            return values.reshape(self.shape), 0.0, None

        try:
            position_axis = POSITION_FIELDS.index(logical)
        except ValueError as exc:
            raise RuntimeError(
                f"Lattice residual requested for non-position field {logical}."
            ) from exc
        digit_axis = self.position_digit_axes[position_axis]
        transformed = np.empty(self.count, dtype=np.float32)
        wrap_offsets = np.empty(self.count, dtype=np.int8)
        roundtrip_error = 0.0
        for start in range(0, self.count, IMPLICIT_CHUNK_ROWS):
            end = min(self.count, start + IMPLICIT_CHUNK_ROWS)
            predictor = self._implicit_predictor(digit_axis, start, end)
            original = values[start:end].astype(np.float64, copy=False)
            unwrapped_residual = original - predictor
            wrapped_residual = np.remainder(
                unwrapped_residual + 0.5,
                1.0,
            ) - 0.5
            chunk_offsets = np.rint(
                unwrapped_residual - wrapped_residual
            )
            if np.any(np.abs(chunk_offsets) > 1):
                raise RuntimeError(
                    "Unexpected periodic wrap offset for lattice field "
                    f"{logical}."
                )
            encoded = wrapped_residual.astype(np.float32)
            offsets = chunk_offsets.astype(np.int8)
            transformed[start:end] = encoded
            wrap_offsets[start:end] = offsets
            restored = (
                encoded.astype(np.float64)
                + predictor
                + offsets.astype(np.float64)
            ).astype(np.float32)
            roundtrip_error = max(
                roundtrip_error,
                float(
                    np.max(
                        np.abs(restored.astype(np.float64) - original),
                        initial=0.0,
                    )
                ),
            )
        return transformed.reshape(self.shape), roundtrip_error, wrap_offsets

    def decode_field(
        self,
        dense_values: np.ndarray,
        logical: str,
        transform: str,
        output_dtype: np.dtype,
        wrap_offsets: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Gather occupied cells and invert an optional position residual."""

        dense = np.asarray(dense_values)
        if dense.size != self.dense_count:
            raise RuntimeError(
                f"Lattice field {logical} expected {self.dense_count} dense "
                f"values, got {dense.size}."
            )
        if self.implicit_full_lattice:
            return self._decode_implicit_full_field(
                dense,
                logical,
                transform,
                output_dtype,
                wrap_offsets,
            )
        gathered = dense.reshape(-1)[self.dense_indices]
        if transform == POSITION_RESIDUAL_TRANSFORM:
            position_axis = POSITION_FIELDS.index(logical)
            digit_axis = self.position_digit_axes[position_axis]
            predictor = (
                self.coordinates[digit_axis].astype(np.float64) / self.side
            )
            if wrap_offsets is None:
                # Compatibility with early experimental manifests that did
                # not preserve the non-periodic side of the domain seam.
                gathered = np.remainder(
                    gathered.astype(np.float64) + predictor,
                    1.0,
                )
            else:
                offsets = np.asarray(wrap_offsets)
                if offsets.ndim != 1 or offsets.size != self.count:
                    raise RuntimeError(
                        f"Lattice wrap offsets for {logical} expected "
                        f"{self.count} values, got shape {offsets.shape}."
                    )
                gathered = (
                    gathered.astype(np.float64)
                    + predictor
                    + offsets.astype(np.float64)
                )
        elif transform != IDENTITY_TRANSFORM:
            raise RuntimeError(
                f"Unsupported lattice field transform for {logical}: {transform}."
            )
        return np.asarray(gathered, dtype=output_dtype)

    def _decode_implicit_full_field(
        self,
        dense_values: np.ndarray,
        logical: str,
        transform: str,
        output_dtype: np.dtype,
        wrap_offsets: Optional[np.ndarray],
    ) -> np.ndarray:
        gathered = dense_values.reshape(-1)
        if transform == IDENTITY_TRANSFORM:
            return np.asarray(gathered, dtype=output_dtype)
        if transform != POSITION_RESIDUAL_TRANSFORM:
            raise RuntimeError(
                f"Unsupported lattice field transform for {logical}: "
                f"{transform}."
            )
        position_axis = POSITION_FIELDS.index(logical)
        digit_axis = self.position_digit_axes[position_axis]
        if wrap_offsets is not None:
            wrap_offsets = np.asarray(wrap_offsets)
            if wrap_offsets.ndim != 1 or wrap_offsets.size != self.count:
                raise RuntimeError(
                    f"Lattice wrap offsets for {logical} expected "
                    f"{self.count} values, got shape {wrap_offsets.shape}."
                )
        decoded = np.empty(self.count, dtype=output_dtype)
        for start in range(0, self.count, IMPLICIT_CHUNK_ROWS):
            end = min(self.count, start + IMPLICIT_CHUNK_ROWS)
            predictor = self._implicit_predictor(digit_axis, start, end)
            values = gathered[start:end].astype(np.float64) + predictor
            if wrap_offsets is None:
                values = np.remainder(values, 1.0)
            else:
                values += wrap_offsets[start:end].astype(np.float64)
            decoded[start:end] = values.astype(output_dtype)
        return decoded

    def _implicit_predictor(
        self,
        digit_axis: int,
        start: int,
        end: int,
    ) -> np.ndarray:
        linear = np.arange(start, end, dtype=np.int64)
        if digit_axis == 0:
            coordinate = linear // (self.side * self.side)
        elif digit_axis == 1:
            coordinate = (linear // self.side) % self.side
        elif digit_axis == 2:
            coordinate = linear % self.side
        else:
            raise RuntimeError(f"Invalid implicit lattice axis: {digit_axis}.")
        return coordinate.astype(np.float64) / self.side


def infer_dense_lattice_layout(
    sorted_ids: np.ndarray,
    sorted_positions: Mapping[str, np.ndarray],
    side: int,
    min_occupancy: float,
) -> DenseLatticeLayout:
    """Infer ID base, position-axis mapping, and periodic dense geometry."""

    ids = np.asarray(sorted_ids)
    if ids.ndim != 1 or ids.size == 0:
        raise LatticeLayoutUnavailable("particle IDs are empty or not one-dimensional")
    if not np.issubdtype(ids.dtype, np.integer):
        raise LatticeLayoutUnavailable("particle IDs are not integers")
    if side <= 0:
        raise LatticeLayoutUnavailable("lattice side must be positive")
    if not 0.0 < min_occupancy <= 1.0:
        raise RuntimeError("Lattice minimum occupancy must be in (0, 1].")
    if np.any(ids[1:] <= ids[:-1]):
        raise LatticeLayoutUnavailable(
            "ID-sorted particles contain duplicate or non-increasing IDs"
        )

    normalized_positions = _validated_positions(sorted_positions, ids.size)
    sample_indices = _sample_indices(ids.size)
    sampled_ids = ids[sample_indices]
    sampled_positions = {
        logical: values[sample_indices]
        for logical, values in normalized_positions.items()
    }
    id_base, position_digit_axes = _infer_id_base_and_axis_mapping(
        sampled_ids,
        sampled_positions,
        side,
        int(ids[0]),
        int(ids[-1]),
    )
    coordinates = decode_lattice_coordinates(ids, side, id_base)
    starts, shape = _periodic_geometry(coordinates, side)
    dense_count = math.prod(shape)
    occupancy = ids.size / dense_count
    if dense_count < MIN_CODEC_VALUES:
        raise LatticeLayoutUnavailable(
            f"dense box has {dense_count} values; at least "
            f"{MIN_CODEC_VALUES} are required by the fieldwise codecs"
        )
    if occupancy < min_occupancy:
        raise LatticeLayoutUnavailable(
            f"best periodic dense box occupancy {occupancy:.6f} is below "
            f"the configured minimum {min_occupancy:.6f}"
        )
    dense_indices = _dense_indices(coordinates, starts, shape, side)
    if np.all(dense_indices[1:] >= dense_indices[:-1]):
        dense_order = np.arange(ids.size, dtype=np.intp)
    else:
        dense_order = np.argsort(dense_indices, kind="stable")
    sorted_dense_indices = dense_indices[dense_order]
    if np.any(sorted_dense_indices[1:] == sorted_dense_indices[:-1]):
        raise LatticeLayoutUnavailable("particle IDs map to duplicate lattice cells")
    occupied = np.zeros(dense_count, dtype=bool)
    occupied[dense_indices] = True
    missing_indices = np.flatnonzero(~occupied)
    return DenseLatticeLayout(
        side=side,
        id_base=id_base,
        starts=starts,
        shape=shape,
        position_digit_axes=position_digit_axes,
        count=int(ids.size),
        dense_indices=dense_indices,
        coordinates=coordinates,
        dense_order=dense_order,
        missing_indices=missing_indices,
    )


def infer_complete_lattice_layout(
    sampled_ids: np.ndarray,
    sampled_positions: Mapping[str, np.ndarray],
    side: int,
    full_minimum: int,
    full_maximum: int,
) -> DenseLatticeLayout:
    """Infer axis mapping for a proven complete lattice without full arrays."""

    count = side**3
    if side <= 0 or full_maximum - full_minimum + 1 != count:
        raise LatticeLayoutUnavailable(
            "merged IDs do not span one complete cubic lattice"
        )
    normalized_positions = _validated_positions(
        sampled_positions,
        int(np.asarray(sampled_ids).size),
    )
    id_base, position_digit_axes = _infer_id_base_and_axis_mapping(
        np.asarray(sampled_ids),
        normalized_positions,
        side,
        full_minimum,
        full_maximum,
    )
    return _complete_lattice_layout(side, id_base, position_digit_axes)


def lattice_layout_from_metadata(
    sorted_ids: Optional[np.ndarray],
    metadata: Mapping[str, Any],
) -> DenseLatticeLayout:
    if metadata.get("name") != LATTICE_LAYOUT_NAME or not metadata.get(
        "enabled", False
    ):
        raise RuntimeError("Manifest does not describe an enabled lattice layout.")
    side = int(metadata["side"])
    id_base = int(metadata["id_base"])
    starts = _triple(metadata["periodic_starts"], "periodic_starts")
    shape = _triple(metadata["dense_shape"], "dense_shape")
    position_digit_axes = _triple(
        metadata["position_digit_axes"],
        "position_digit_axes",
    )
    if side <= 0:
        raise RuntimeError(f"Invalid lattice side: {side}.")
    if id_base not in (0, 1):
        raise RuntimeError(f"Invalid lattice ID base: {id_base}.")
    if any(start < 0 or start >= side for start in starts):
        raise RuntimeError(f"Invalid lattice periodic starts: {starts!r}.")
    if any(value <= 0 or value > side for value in shape):
        raise RuntimeError(f"Invalid lattice dense shape: {shape!r}.")
    if sorted(position_digit_axes) != [0, 1, 2]:
        raise RuntimeError(
            "Lattice position_digit_axes must be a permutation of (0, 1, 2)."
        )
    if int(metadata["dense_count"]) != math.prod(shape):
        raise RuntimeError("Lattice dense_count does not match dense_shape.")
    expected_count = int(metadata["particle_count"])
    if bool(metadata.get("implicit_full_lattice", False)):
        if starts != (0, 0, 0) or shape != (side, side, side):
            raise RuntimeError(
                "Implicit full lattice metadata must describe the full cube."
            )
        if expected_count != side**3:
            raise RuntimeError(
                "Implicit full lattice particle count does not match side^3."
            )
        if sorted_ids is not None:
            ids = np.asarray(sorted_ids)
            if ids.size != expected_count:
                raise RuntimeError(
                    f"Lattice manifest expected {expected_count} IDs, "
                    f"got {ids.size}."
                )
            if ids.size and (
                int(ids[0]) != id_base
                or int(ids[-1]) != id_base + expected_count - 1
            ):
                raise RuntimeError(
                    "Implicit full lattice IDs do not match its recorded base."
                )
        return _complete_lattice_layout(
            side,
            id_base,
            position_digit_axes,
        )
    if sorted_ids is None:
        raise RuntimeError("Explicit lattice reconstruction requires IDs.")
    ids = np.asarray(sorted_ids)
    if ids.size != expected_count:
        raise RuntimeError(
            f"Lattice manifest expected {expected_count} IDs, got {ids.size}."
        )
    coordinates = decode_lattice_coordinates(ids, side, id_base)
    dense_indices = _dense_indices(coordinates, starts, shape, side)
    return DenseLatticeLayout(
        side=side,
        id_base=id_base,
        starts=starts,
        shape=shape,
        position_digit_axes=position_digit_axes,
        count=expected_count,
        dense_indices=dense_indices,
        coordinates=coordinates,
        dense_order=np.empty(0, dtype=np.intp),
        missing_indices=np.empty(0, dtype=np.intp),
    )


def _complete_lattice_layout(
    side: int,
    id_base: int,
    position_digit_axes: Tuple[int, int, int],
) -> DenseLatticeLayout:
    count = side**3
    empty = np.empty(0, dtype=np.intp)
    empty_coordinates = (
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
    )
    return DenseLatticeLayout(
        side=side,
        id_base=id_base,
        starts=(0, 0, 0),
        shape=(side, side, side),
        position_digit_axes=position_digit_axes,
        count=count,
        dense_indices=empty,
        coordinates=empty_coordinates,
        dense_order=empty,
        missing_indices=empty,
        implicit_full_lattice=True,
    )


def decode_lattice_coordinates(
    ids: np.ndarray,
    side: int,
    id_base: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(ids)
    if ids.ndim != 1 or ids.size == 0:
        raise LatticeLayoutUnavailable(
            "particle IDs are empty or not one-dimensional"
        )
    if not np.issubdtype(ids.dtype, np.integer):
        raise LatticeLayoutUnavailable("particle IDs are not integers")
    minimum = int(ids.min())
    maximum = int(ids.max())
    total = side**3
    if minimum < id_base or maximum >= id_base + total:
        raise LatticeLayoutUnavailable(
            f"IDs [{minimum}, {maximum}] do not fit base {id_base} "
            f"lattice with {total} cells"
        )
    linear = ids.astype(np.uint64, copy=False) - np.uint64(id_base)
    plane = np.uint64(side * side)
    high = (linear // plane).astype(np.int32)
    middle = ((linear // np.uint64(side)) % side).astype(np.int32)
    low = (linear % side).astype(np.int32)
    return high, middle, low


def position_transform_guard(values: np.ndarray, measured_error: float) -> float:
    """Conservative float32 addition/rounding budget for residual inversion."""

    magnitude = max(
        1.0,
        float(np.max(np.abs(values.astype(np.float64)), initial=0.0)),
    )
    return max(
        float(measured_error),
        2.0 * float(np.finfo(np.float32).eps) * magnitude,
    )


def _validated_positions(
    positions: Mapping[str, np.ndarray],
    count: int,
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for logical in POSITION_FIELDS:
        if logical not in positions:
            raise LatticeLayoutUnavailable(f"missing normalized position {logical}")
        values = np.asarray(positions[logical], dtype=np.float32)
        if values.ndim != 1 or values.size != count:
            raise LatticeLayoutUnavailable(
                f"normalized position {logical} has shape {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise LatticeLayoutUnavailable(
                f"normalized position {logical} contains non-finite values"
            )
        minimum = float(values.min())
        maximum = float(values.max())
        if minimum < 0.0 or maximum >= 1.0:
            raise LatticeLayoutUnavailable(
                f"normalized position {logical} range [{minimum}, {maximum}] "
                "is outside the periodic [0, 1) domain"
            )
        result[logical] = values
    return result


def _sample_indices(count: int, maximum: int = 200_000) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.intp)
    return np.linspace(0, count - 1, maximum, dtype=np.intp)


def _infer_id_base_and_axis_mapping(
    sampled_ids: np.ndarray,
    sampled_positions: Mapping[str, np.ndarray],
    side: int,
    full_minimum: int,
    full_maximum: int,
) -> Tuple[int, Tuple[int, int, int]]:
    candidates = [
        base
        for base in (0, 1)
        if full_minimum >= base and full_maximum < base + side**3
    ]
    if not candidates:
        raise LatticeLayoutUnavailable(
            "particle IDs fit neither a zero-based nor one-based lattice"
        )
    best: Optional[Tuple[float, int, Tuple[int, int, int]]] = None
    for base in candidates:
        coordinates = decode_lattice_coordinates(sampled_ids, side, base)
        for permutation in itertools.permutations(range(3)):
            score = 0.0
            for position_axis, logical in enumerate(POSITION_FIELDS):
                mesh_position = (
                    sampled_positions[logical].astype(np.float64) * side
                )
                coordinate = coordinates[permutation[position_axis]].astype(
                    np.float64
                )
                displacement = np.remainder(
                    mesh_position - coordinate + side / 2.0,
                    side,
                ) - side / 2.0
                score += float(np.mean(displacement * displacement))
            candidate = (score, base, tuple(int(value) for value in permutation))
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best[1], best[2]


def _periodic_geometry(
    coordinates: Sequence[np.ndarray],
    side: int,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    starts = []
    spans = []
    for values in coordinates:
        present_mask = np.zeros(side, dtype=bool)
        present_mask[values] = True
        present = np.flatnonzero(present_mask)
        if present.size == side:
            starts.append(0)
            spans.append(side)
            continue
        following = np.roll(present, -1)
        gaps = (following - present) % side
        cut = int(np.argmax(gaps))
        start = int(following[cut])
        unwrapped = (present - start) % side
        starts.append(start)
        spans.append(int(unwrapped.max()) + 1)
    return tuple(starts), tuple(spans)


def _dense_indices(
    coordinates: Sequence[np.ndarray],
    starts: Tuple[int, int, int],
    shape: Tuple[int, int, int],
    side: int,
) -> np.ndarray:
    locals_by_axis = [
        (coordinates[axis].astype(np.int64) - starts[axis]) % side
        for axis in range(3)
    ]
    for axis, local in enumerate(locals_by_axis):
        if int(local.max(initial=0)) >= shape[axis]:
            raise RuntimeError(
                f"Lattice coordinate exceeds stored dense axis {axis}."
            )
    indices = (
        (locals_by_axis[0] * shape[1] + locals_by_axis[1]) * shape[2]
        + locals_by_axis[2]
    )
    return indices.astype(np.intp, copy=False)


def _triple(values: Any, label: str) -> Tuple[int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 3 or any(value < 0 for value in result):
        raise RuntimeError(f"Invalid lattice {label}: {values!r}.")
    return result  # type: ignore[return-value]


__all__ = [
    "DenseLatticeLayout",
    "IDENTITY_TRANSFORM",
    "LATTICE_LAYOUT_NAME",
    "LatticeLayoutUnavailable",
    "POSITION_RESIDUAL_TRANSFORM",
    "decode_lattice_coordinates",
    "infer_complete_lattice_layout",
    "infer_dense_lattice_layout",
    "lattice_layout_from_metadata",
    "position_transform_guard",
]
