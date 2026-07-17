"""Small immutable models shared by pipeline stages."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class ToolPaths:
    """Paths to native executables used by the pipeline."""

    lcp: Path


@dataclass(frozen=True)
class PositionScale:
    """Conversion from stored position units to compressor units."""

    mode: str
    value: float
    attr: Optional[str] = None


@dataclass(frozen=True)
class ErrorBoundSelection:
    """Resolved absolute bounds and their user-facing selection mode."""

    mode: str
    abs_by_field: Dict[str, float]
    relative: Optional[float] = None
    compressor_abs: Optional[float] = None


@dataclass(frozen=True)
class CanonicalOrder:
    """The row order selected by whichever triplet is compressed with LCP."""

    mapping: str = "original_row"
    field: Optional[str] = None
    artifact: Optional[str] = None
    values: Optional[np.ndarray] = None

    @property
    def is_reordered(self) -> bool:
        return self.values is not None

