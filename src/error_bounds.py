"""Resolve user error-bound options into per-field compressor bounds."""

import argparse
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import h5py
import numpy as np

from src.constants import POSITION_FIELDS, VELOCITY_FIELDS
from src.models import ErrorBoundSelection, PositionScale


@dataclass(frozen=True)
class ResolvedErrorBounds:
    position: ErrorBoundSelection
    velocity: ErrorBoundSelection
    position_lcp_abs: float
    position_vector_abs: float
    velocity_abs: float
    id_abs: float
    fields: Dict[str, Dict[str, Any]]


def validate_error_bound(value: float, label: str) -> float:
    value = float(value)
    if value < 0.0:
        raise RuntimeError(f"{label} must be non-negative.")
    return value


def select_relative_or_absolute(
    args: argparse.Namespace,
    prefix: str,
    fields: Iterable[str],
    ranges: Mapping[str, float],
    default_abs: float,
    compressor_abs: Optional[float] = None,
) -> ErrorBoundSelection:
    specific_relative = getattr(args, f"{prefix}_rel_eb")
    specific_absolute = getattr(args, f"{prefix}_abs_eb")
    option_prefix = prefix.replace("_", "-")
    if specific_relative is not None and specific_absolute is not None:
        raise RuntimeError(
            f"--{option_prefix}-rel-eb and --{option_prefix}-abs-eb "
            "cannot both be set."
        )
    if specific_relative is not None:
        relative = validate_error_bound(
            specific_relative,
            f"--{option_prefix}-rel-eb",
        )
        return ErrorBoundSelection(
            "relative",
            {field: relative * float(ranges[field]) for field in fields},
            relative=relative,
            compressor_abs=compressor_abs,
        )
    if specific_absolute is not None:
        absolute = validate_error_bound(
            specific_absolute,
            f"--{option_prefix}-abs-eb",
        )
        return ErrorBoundSelection(
            "absolute",
            {field: absolute for field in fields},
            compressor_abs=compressor_abs,
        )
    if args.rel_eb is not None:
        relative = validate_error_bound(args.rel_eb, "--rel-eb")
        return ErrorBoundSelection(
            "relative",
            {field: relative * float(ranges[field]) for field in fields},
            relative=relative,
            compressor_abs=compressor_abs,
        )
    absolute = validate_error_bound(default_abs, "--abs-eb")
    return ErrorBoundSelection(
        "absolute",
        {field: absolute for field in fields},
        compressor_abs=compressor_abs,
    )


def serialize_error_bound_selection(
    selection: ErrorBoundSelection,
    fields: Iterable[str],
    ranges: Mapping[str, float],
    range_units: str,
) -> Dict[str, Dict[str, Any]]:
    return {
        field: {
            "mode": selection.mode,
            "abs": float(selection.abs_by_field[field]),
            "relative": selection.relative,
            "range": float(ranges[field]),
            "range_units": range_units,
            "compressor_abs": float(
                selection.compressor_abs
                if selection.compressor_abs is not None
                else selection.abs_by_field[field]
            ),
        }
        for field in fields
    }


def resolve_error_bounds(
    args: argparse.Namespace,
    h5: h5py.File,
    fields: Mapping[str, str],
    position_scale: PositionScale,
    statistics: Mapping[str, Any],
) -> ResolvedErrorBounds:
    position_stats = statistics["positions"]
    velocity_stats = statistics["velocities"]
    position_ranges = {
        field: float(position_stats[field]["range_in_lcp_units"])
        for field in POSITION_FIELDS
    }
    velocity_ranges = {
        field: float(velocity_stats[field]["float_range"])
        for field in VELOCITY_FIELDS
    }
    position_diagonal = math.sqrt(
        sum(value * value for value in position_ranges.values())
    )

    base_position = select_relative_or_absolute(
        args,
        "pos",
        POSITION_FIELDS,
        position_ranges,
        args.abs_eb,
    )
    position_preprocess_errors = {
        field: _position_preprocess_error(
            h5,
            fields[field],
            position_stats[field],
            position_scale,
        )
        for field in POSITION_FIELDS
    }
    (
        position_lcp_abs,
        position_vector_abs,
        vector_requested_abs,
        vector_preprocess_error,
    ) = _resolve_position_compressor_bounds(
        base_position,
        position_preprocess_errors,
        position_diagonal,
    )
    position = ErrorBoundSelection(
        base_position.mode,
        base_position.abs_by_field,
        relative=base_position.relative,
        compressor_abs=position_lcp_abs,
    )

    base_velocity = select_relative_or_absolute(
        args,
        "vel",
        VELOCITY_FIELDS,
        velocity_ranges,
        args.abs_eb,
    )
    if args.vel_compressor == "lcp":
        velocity_abs = min(
            max(
                0.0,
                base_velocity.abs_by_field[field]
                - float(
                    velocity_stats[field]["preprocess_cast_max_abs"]
                ),
            )
            for field in VELOCITY_FIELDS
        )
        velocity = ErrorBoundSelection(
            base_velocity.mode,
            base_velocity.abs_by_field,
            relative=base_velocity.relative,
            compressor_abs=float(velocity_abs),
        )
    else:
        velocity_abs = max(base_velocity.abs_by_field.values())
        velocity = base_velocity

    id_abs = validate_error_bound(args.id_abs_eb, "--id-abs-eb")
    field_bounds = serialize_error_bound_selection(
        position,
        POSITION_FIELDS,
        position_ranges,
        "lcp_units",
    )
    field_bounds["positions_xnyzip"] = {
        "mode": base_position.mode,
        "abs": vector_requested_abs,
        "relative": base_position.relative,
        "range": position_diagonal,
        "range_units": "lcp_units_bbox_diagonal",
        "compressor_abs": position_vector_abs,
        "preprocess_l2_max_abs": vector_preprocess_error,
    }
    field_bounds.update(
        serialize_error_bound_selection(
            velocity,
            VELOCITY_FIELDS,
            velocity_ranges,
            "source_units",
        )
    )
    id_stats = statistics["id"]
    field_bounds["id"] = {
        "mode": "lossless",
        "abs": id_abs,
        "relative": None,
        "range": (
            float(id_stats["max"] - id_stats["min"])
            if id_stats["min"] is not None
            else None
        ),
        "range_units": "source_units",
        "compressor_abs": 0.0,
    }
    return ResolvedErrorBounds(
        position=position,
        velocity=velocity,
        position_lcp_abs=float(position_lcp_abs),
        position_vector_abs=float(position_vector_abs),
        velocity_abs=float(velocity_abs),
        id_abs=id_abs,
        fields=field_bounds,
    )


def _position_preprocess_error(
    h5: h5py.File,
    dataset_path: str,
    statistics: Mapping[str, Any],
    scale: PositionScale,
) -> float:
    dtype = np.dtype(h5[dataset_path].dtype)
    rounding = 0.5 / scale.value if np.issubdtype(dtype, np.integer) else 0.0
    cast = float(statistics["preprocess_cast_max_abs_in_lcp_units"])
    return cast + rounding


def _resolve_position_compressor_bounds(
    selection: ErrorBoundSelection,
    preprocess_errors: Mapping[str, float],
    position_diagonal: float,
) -> Tuple[float, float, float, float]:
    if selection.mode != "relative":
        absolute = float(min(selection.abs_by_field.values()))
        return absolute, absolute, absolute, 0.0

    compressor_abs = min(
        max(
            0.0,
            selection.abs_by_field[field] - preprocess_errors[field],
        )
        for field in POSITION_FIELDS
    )
    requested_vector_abs = float(
        selection.relative * position_diagonal
    )
    vector_preprocess_error = math.sqrt(
        sum(error * error for error in preprocess_errors.values())
    )
    vector_abs = max(
        0.0,
        requested_vector_abs - vector_preprocess_error,
    )
    return (
        float(compressor_abs),
        float(vector_abs),
        requested_vector_abs,
        vector_preprocess_error,
    )

