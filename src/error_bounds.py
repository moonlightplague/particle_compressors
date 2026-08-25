"""Resolve user error-bound options into per-field compressor bounds."""

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

import h5py
import numpy as np

from src.constants import POSITION_FIELDS, VELOCITY_FIELDS
from src.models import ErrorBoundSelection, PositionScale


@dataclass(frozen=True)
class ResolvedErrorBounds:
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
        )
    if specific_absolute is not None:
        absolute = validate_error_bound(
            specific_absolute,
            f"--{option_prefix}-abs-eb",
        )
        return ErrorBoundSelection(
            "absolute",
            {field: absolute for field in fields},
        )
    if args.rel_eb is not None:
        relative = validate_error_bound(args.rel_eb, "--rel-eb")
        return ErrorBoundSelection(
            "relative",
            {field: relative * float(ranges[field]) for field in fields},
            relative=relative,
        )
    absolute = validate_error_bound(default_abs, "--abs-eb")
    return ErrorBoundSelection(
        "absolute",
        {field: absolute for field in fields},
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
            "compressor_abs": float(selection.abs_by_field[field]),
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
        field: float(position_stats[field]["range_in_compressor_units"])
        for field in POSITION_FIELDS
    }
    velocity_ranges = {
        field: float(velocity_stats[field]["float_range"])
        for field in VELOCITY_FIELDS
    }

    position = select_relative_or_absolute(
        args,
        "pos",
        POSITION_FIELDS,
        position_ranges,
        args.abs_eb,
    )
    position_bounds = serialize_error_bound_selection(
        position,
        POSITION_FIELDS,
        position_ranges,
        "compressor_units",
    )
    if position.mode == "relative":
        for field in POSITION_FIELDS:
            requested = float(position.abs_by_field[field])
            position_bounds[field]["compressor_abs"] = max(
                0.0,
                requested
                - _position_preprocess_error(
                    h5,
                    fields[field],
                    position_stats[field],
                    position_scale,
                ),
            )

    velocity = select_relative_or_absolute(
        args,
        "vel",
        VELOCITY_FIELDS,
        velocity_ranges,
        args.abs_eb,
    )
    field_bounds = {
        **position_bounds,
        **serialize_error_bound_selection(
            velocity,
            VELOCITY_FIELDS,
            velocity_ranges,
            "source_units",
        ),
    }

    id_abs = validate_error_bound(args.id_abs_eb, "--id-abs-eb")
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
    return ResolvedErrorBounds(fields=field_bounds)


def _position_preprocess_error(
    h5: h5py.File,
    dataset_path: str,
    statistics: Mapping[str, Any],
    scale: PositionScale,
) -> float:
    dtype = np.dtype(h5[dataset_path].dtype)
    rounding = 0.5 / scale.value if np.issubdtype(dtype, np.integer) else 0.0
    cast = float(
        statistics["preprocess_cast_max_abs_in_compressor_units"]
    )
    return cast + rounding
