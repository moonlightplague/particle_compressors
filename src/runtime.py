"""Filesystem, subprocess, and optional dependency utilities."""

import ctypes
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from src.models import ToolPaths


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def require_output_path(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise RuntimeError(
            f"{path} already exists. Use --force to overwrite pipeline outputs."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Mapping[str, Any], force: bool = True) -> None:
    require_output_path(path, force)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def json_size_bytes(payload: Mapping[str, Any]) -> int:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return len(rendered.encode("utf-8"))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(argv: List[str]) -> None:
    process = subprocess.run(argv, text=True, capture_output=True)
    if process.returncode == 0:
        return
    raise RuntimeError(
        "Command failed with exit code "
        f"{process.returncode}: {' '.join(argv)}\n"
        f"stdout:\n{process.stdout}\n"
        f"stderr:\n{process.stderr}"
    )


def load_pcodec() -> Tuple[Any, Any]:
    try:
        from pcodec import ChunkConfig, standalone

        return standalone, ChunkConfig
    except ImportError as first_error:
        pco_python = repo_root() / "tools" / "pcodec" / "pco_python"
        if pco_python.is_dir() and str(pco_python) not in sys.path:
            sys.path.insert(0, str(pco_python))
        try:
            from pcodec import ChunkConfig, standalone

            return standalone, ChunkConfig
        except ImportError as second_error:
            raise RuntimeError(
                "Could not import pcodec. Build/install the Python extension "
                "from the submodule with `python -m pip install -e "
                "tools/pcodec/pco_python`."
            ) from second_error or first_error


def load_pysz() -> Tuple[Any, Any, Any]:
    try:
        from pysz import sz, szConfig, szErrorBoundMode

        return sz, szConfig, szErrorBoundMode
    except ImportError as exc:
        raise RuntimeError(
            "Could not import pysz. Install it with `python -m pip install pysz`."
        ) from exc


def load_pyszo() -> Tuple[Any, Any, Any, Any]:
    try:
        _preload_pyszo_zstd()
        from pyszo import szo, szoAlgorithm, szoConfig, szoErrorBoundMode

        return szo, szoConfig, szoErrorBoundMode, szoAlgorithm
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Could not import pyszo. Build/install the local binding with "
            "`python -m pip install -e tools/SZo/tools/pyszo`."
        ) from exc


def _preload_pyszo_zstd() -> None:
    spec = importlib.util.find_spec("pyszo")
    if spec is None or not spec.submodule_search_locations:
        return
    package_dir = Path(next(iter(spec.submodule_search_locations)))
    for library_name in ("libzstd.so", "libzstd.dylib"):
        bundled_zstd = package_dir / library_name
        if bundled_zstd.is_file():
            ctypes.CDLL(str(bundled_zstd), mode=ctypes.RTLD_GLOBAL)
            return


def read_raw(path: str, dtype: np.dtype, count: int) -> np.ndarray:
    data = np.fromfile(path, dtype=dtype, count=count)
    if data.size != count:
        raise RuntimeError(
            f"Unexpected EOF reading {path}; expected {count}, got {data.size}."
        )
    return data


def resolve_velocity_chunk_workers(configured: int) -> int:
    configured = int(configured)
    if configured < 0:
        raise RuntimeError("Velocity chunk workers must be non-negative.")
    if configured:
        return configured
    return min(16, os.cpu_count() or 1)


def resolve_lcp_chunk_workers(configured: int) -> int:
    """Compatibility alias for the codec-neutral worker resolver."""

    return resolve_velocity_chunk_workers(configured)


__all__ = [
    "ToolPaths",
    "json_size_bytes",
    "load_pcodec",
    "load_pysz",
    "load_pyszo",
    "read_json",
    "read_raw",
    "repo_root",
    "require_output_path",
    "resolve_lcp_chunk_workers",
    "resolve_velocity_chunk_workers",
    "run_command",
    "write_json",
]
