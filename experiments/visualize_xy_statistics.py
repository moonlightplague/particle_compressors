"""Visualize particle statistics projected onto the x-y plane.

The input may be one or more HDF5 files or directories containing HDF5
files.  Rows from every input are accumulated into a common x-y grid and a
self-contained HTML dashboard is written without requiring a plotting
package.  Integer positions are divided by the HDF5 ``bitwidth`` attribute by
default, matching the compressor's position-unit convention.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hdf5_io import resolve_fields  # noqa: E402


SUMMARY_FIELDS = ("x", "y", "z", "vx", "vy", "vz", "speed")
GRID_FIELDS = (
    "particle_count",
    "mean_z",
    "mean_speed",
    "velocity_dispersion",
    "mean_vx",
    "mean_vy",
    "mean_vz",
)


@dataclass(frozen=True)
class InputMetadata:
    path: Path
    count: int
    fields: dict[str, str]
    position_scale: float
    rank: Optional[int]
    nsidemesh: Optional[int]


class ScalarSummary:
    """Accumulate a stable-enough float64 summary without retaining rows."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values: np.ndarray) -> None:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            return
        self.count += int(finite.size)
        self.total += float(np.sum(finite, dtype=np.float64))
        self.total_square += float(
            np.sum(np.square(finite), dtype=np.float64)
        )
        self.minimum = min(self.minimum, float(np.min(finite)))
        self.maximum = max(self.maximum, float(np.max(finite)))

    def payload(self) -> dict[str, Any]:
        if not self.count:
            return {
                "count": 0,
                "minimum": None,
                "maximum": None,
                "mean": None,
                "standard_deviation": None,
            }
        mean = self.total / self.count
        variance = max(0.0, self.total_square / self.count - mean * mean)
        return {
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": mean,
            "standard_deviation": math.sqrt(variance),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help=(
            "HDF5 files and/or directories. Direct .h5 children of each "
            "directory are included."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("xy_statistics.html"),
        help="Self-contained HTML output (default: xy_statistics.html).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=256,
        help="Number of bins along each x-y axis (default: 256).",
    )
    parser.add_argument(
        "--max-particles",
        type=int,
        default=5_000_000,
        help=(
            "Maximum rows analyzed via deterministic systematic sampling; "
            "use 0 to analyze every row (default: 5000000)."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4_000_000,
        help="Maximum source-row span processed at once (default: 4000000).",
    )
    parser.add_argument(
        "--position-scale",
        default="auto",
        metavar="AUTO|RAW|VALUE",
        help=(
            "Position divisor: auto uses bitwidth for integer positions, "
            "raw uses 1, or supply a positive numeric value."
        ),
    )
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Fixed plotted extent; otherwise it is inferred from analyzed rows.",
    )
    return parser.parse_args()


def build_xy_statistics(
    inputs: Sequence[Path],
    output: Path,
    *,
    bins: int = 256,
    max_particles: int = 5_000_000,
    chunk_size: int = 4_000_000,
    position_scale: str = "auto",
    extent: Optional[Sequence[float]] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Build the dashboard and return the JSON-compatible plot payload."""
    paths = discover_h5_files(inputs)
    if not paths:
        raise RuntimeError("No .h5 input files were found.")
    if bins < 2 or bins > 1024:
        raise RuntimeError("--bins must be between 2 and 1024.")
    if max_particles < 0:
        raise RuntimeError("--max-particles cannot be negative.")
    if chunk_size <= 0:
        raise RuntimeError("--chunk-size must be positive.")

    metadata = [inspect_input(path, position_scale) for path in paths]
    total_count = sum(item.count for item in metadata)
    if total_count <= 0:
        raise RuntimeError("The selected HDF5 files contain no particle rows.")
    stride = (
        1
        if max_particles == 0 or max_particles >= total_count
        else math.ceil(total_count / max_particles)
    )
    analyzed_count = (total_count - 1) // stride + 1

    if extent is None:
        plot_extent = infer_extent(metadata, stride, chunk_size, verbose)
    else:
        plot_extent = _validate_extent(extent)

    grid_size = bins * bins
    count_grid = np.zeros(grid_size, dtype=np.int64)
    sums = {
        name: np.zeros(grid_size, dtype=np.float64)
        for name in ("z", "vx", "vy", "vz", "speed", "speed_square")
    }
    summaries = {name: ScalarSummary() for name in SUMMARY_FIELDS}

    global_offset = 0
    rows_seen = 0
    for file_index, item in enumerate(metadata, start=1):
        if verbose:
            print(
                f"accumulating {file_index}/{len(metadata)}: "
                f"{item.path.name}",
                flush=True,
            )
        with h5py.File(item.path, "r") as source:
            for selection in _sample_slices(
                item.count,
                global_offset,
                stride,
                chunk_size,
            ):
                arrays = _read_fields(source, item, selection)
                rows_seen += int(arrays["x"].size)
                speed_square = (
                    arrays["vx"] * arrays["vx"]
                    + arrays["vy"] * arrays["vy"]
                    + arrays["vz"] * arrays["vz"]
                )
                speed = np.sqrt(speed_square)
                linear = _bin_indices(
                    arrays["x"],
                    arrays["y"],
                    plot_extent,
                    bins,
                )
                count_grid += np.bincount(
                    linear,
                    minlength=grid_size,
                ).astype(np.int64, copy=False)
                for name, values in (
                    ("z", arrays["z"]),
                    ("vx", arrays["vx"]),
                    ("vy", arrays["vy"]),
                    ("vz", arrays["vz"]),
                    ("speed", speed),
                    ("speed_square", speed_square),
                ):
                    sums[name] += np.bincount(
                        linear,
                        weights=values,
                        minlength=grid_size,
                    )
                for name in ("x", "y", "z", "vx", "vy", "vz"):
                    summaries[name].update(arrays[name])
                summaries["speed"].update(speed)
        global_offset += item.count

    if rows_seen != analyzed_count:
        raise RuntimeError(
            f"Sampling selected {rows_seen} rows; expected {analyzed_count}."
        )

    occupied = count_grid > 0
    means = {
        name: _masked_mean(values, count_grid, occupied)
        for name, values in sums.items()
    }
    mean_speed_square = means["speed_square"]
    velocity_dispersion = np.full(grid_size, np.nan, dtype=np.float64)
    velocity_dispersion[occupied] = np.sqrt(
        np.maximum(
            0.0,
            mean_speed_square[occupied]
            - means["vx"][occupied] ** 2
            - means["vy"][occupied] ** 2
            - means["vz"][occupied] ** 2,
        )
    )

    grid_payload = {
        "particle_count": count_grid.tolist(),
        "mean_z": _nullable_list(means["z"]),
        "mean_speed": _nullable_list(means["speed"]),
        "velocity_dispersion": _nullable_list(velocity_dispersion),
        "mean_vx": _nullable_list(means["vx"]),
        "mean_vy": _nullable_list(means["vy"]),
        "mean_vz": _nullable_list(means["vz"]),
    }
    input_payload = [
        {
            "path": str(item.path),
            "name": item.path.name,
            "count": item.count,
            "position_scale": item.position_scale,
            "rank": item.rank,
            "nsidemesh": item.nsidemesh,
        }
        for item in metadata
    ]
    payload: dict[str, Any] = {
        "title": "Particle statistics on the x-y plane",
        "inputs": input_payload,
        "file_count": len(metadata),
        "total_count": total_count,
        "analyzed_count": analyzed_count,
        "sampling_stride": stride,
        "sampling_is_exact": stride == 1,
        "bins": bins,
        "extent": {
            "x_min": plot_extent[0],
            "x_max": plot_extent[1],
            "y_min": plot_extent[2],
            "y_max": plot_extent[3],
        },
        "position_units": "source position divided by position_scale",
        "summaries": {
            name: summary.payload() for name, summary in summaries.items()
        },
        "grid_fields": list(GRID_FIELDS),
        "grids": grid_payload,
    }

    output_path = output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embedded = json.dumps(payload, separators=(",", ":")).replace(
        "</",
        "<\\/",
    )
    output_path.write_text(
        HTML_TEMPLATE.replace("__EMBEDDED_DATA__", embedded),
        encoding="utf-8",
    )
    payload["output"] = str(output_path)
    return payload


def discover_h5_files(inputs: Sequence[Path]) -> list[Path]:
    """Resolve files and direct directory children, preserving stable order."""
    discovered: list[Path] = []
    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        if path.is_dir():
            discovered.extend(
                sorted(
                    child.resolve()
                    for child in path.iterdir()
                    if child.is_file() and child.suffix == ".h5"
                )
            )
        elif path.is_file():
            if path.suffix != ".h5":
                raise RuntimeError(f"Input is not an .h5 file: {path}")
            discovered.append(path)
        else:
            raise RuntimeError(f"Input does not exist: {path}")
    return list(dict.fromkeys(discovered))


def inspect_input(path: Path, position_scale: str) -> InputMetadata:
    with h5py.File(path, "r") as source:
        fields = resolve_fields(source)
        shapes = {logical: source[field].shape for logical, field in fields.items()}
        if any(len(shape) != 1 for shape in shapes.values()):
            raise RuntimeError(f"All particle datasets must be one-dimensional: {path}")
        counts = {int(shape[0]) for shape in shapes.values()}
        if len(counts) != 1:
            raise RuntimeError(f"Particle dataset lengths do not match: {path}")
        scale = _resolve_position_scale(source, fields, position_scale, path)
        return InputMetadata(
            path=path,
            count=counts.pop(),
            fields=fields,
            position_scale=scale,
            rank=_optional_int_attribute(source, "rank"),
            nsidemesh=_optional_int_attribute(source, "nsidemesh"),
        )


def _resolve_position_scale(
    source: h5py.File,
    fields: dict[str, str],
    requested: str,
    path: Path,
) -> float:
    normalized = requested.strip().lower()
    if normalized == "raw":
        return 1.0
    if normalized == "auto":
        position_dtypes = [source[fields[name]].dtype for name in ("x", "y", "z")]
        if all(np.issubdtype(dtype, np.integer) for dtype in position_dtypes):
            if "bitwidth" not in source.attrs:
                raise RuntimeError(
                    f"Integer positions require the bitwidth attribute in auto "
                    f"mode: {path}. Use --position-scale raw or VALUE."
                )
            value = float(source.attrs["bitwidth"])
        else:
            value = 1.0
    else:
        try:
            value = float(requested)
        except ValueError as error:
            raise RuntimeError(
                "--position-scale must be auto, raw, or a positive number."
            ) from error
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"Invalid position scale {value!r} for {path}.")
    return value


def _optional_int_attribute(source: h5py.File, name: str) -> Optional[int]:
    if name not in source.attrs:
        return None
    return int(source.attrs[name])


def infer_extent(
    metadata: Sequence[InputMetadata],
    stride: int,
    chunk_size: int,
    verbose: bool,
) -> tuple[float, float, float, float]:
    x_min, x_max = math.inf, -math.inf
    y_min, y_max = math.inf, -math.inf
    global_offset = 0
    for file_index, item in enumerate(metadata, start=1):
        if verbose:
            print(
                f"scanning extent {file_index}/{len(metadata)}: "
                f"{item.path.name}",
                flush=True,
            )
        with h5py.File(item.path, "r") as source:
            for selection in _sample_slices(
                item.count,
                global_offset,
                stride,
                chunk_size,
            ):
                x = (
                    np.asarray(source[item.fields["x"]][selection], dtype=np.float64)
                    / item.position_scale
                )
                y = (
                    np.asarray(source[item.fields["y"]][selection], dtype=np.float64)
                    / item.position_scale
                )
                if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                    raise RuntimeError(
                        f"Non-finite x or y position encountered in {item.path}."
                    )
                if x.size:
                    x_min = min(x_min, float(np.min(x)))
                    x_max = max(x_max, float(np.max(x)))
                    y_min = min(y_min, float(np.min(y)))
                    y_max = max(y_max, float(np.max(y)))
        global_offset += item.count
    return _validate_extent((x_min, x_max, y_min, y_max), pad_degenerate=True)


def _validate_extent(
    extent: Sequence[float],
    *,
    pad_degenerate: bool = False,
) -> tuple[float, float, float, float]:
    if len(extent) != 4:
        raise RuntimeError("Extent must contain x_min x_max y_min y_max.")
    x_min, x_max, y_min, y_max = (float(value) for value in extent)
    if not all(math.isfinite(value) for value in (x_min, x_max, y_min, y_max)):
        raise RuntimeError("Extent values must be finite.")
    if pad_degenerate:
        x_min, x_max = _pad_degenerate_range(x_min, x_max)
        y_min, y_max = _pad_degenerate_range(y_min, y_max)
    if x_min >= x_max or y_min >= y_max:
        raise RuntimeError("Extent requires XMIN < XMAX and YMIN < YMAX.")
    return x_min, x_max, y_min, y_max


def _pad_degenerate_range(lower: float, upper: float) -> tuple[float, float]:
    if lower != upper:
        return lower, upper
    padding = max(1.0, abs(lower)) * 0.5
    return lower - padding, upper + padding


def _sample_slices(
    count: int,
    global_offset: int,
    stride: int,
    chunk_size: int,
) -> Iterable[slice]:
    """Yield strided slices aligned to one global systematic sample."""
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        first = start + (-(global_offset + start)) % stride
        if first < stop:
            yield slice(first, stop, stride)


def _read_fields(
    source: h5py.File,
    item: InputMetadata,
    selection: slice,
) -> dict[str, np.ndarray]:
    arrays = {
        name: np.asarray(source[item.fields[name]][selection], dtype=np.float64)
        for name in ("x", "y", "z", "vx", "vy", "vz")
    }
    for name in ("x", "y", "z"):
        arrays[name] /= item.position_scale
    if any(not np.all(np.isfinite(values)) for values in arrays.values()):
        raise RuntimeError(f"Non-finite particle value encountered in {item.path}.")
    return arrays


def _bin_indices(
    x: np.ndarray,
    y: np.ndarray,
    extent: tuple[float, float, float, float],
    bins: int,
) -> np.ndarray:
    x_min, x_max, y_min, y_max = extent
    ix = np.floor((x - x_min) * bins / (x_max - x_min)).astype(np.int64)
    iy = np.floor((y - y_min) * bins / (y_max - y_min)).astype(np.int64)
    np.clip(ix, 0, bins - 1, out=ix)
    np.clip(iy, 0, bins - 1, out=iy)
    return iy * bins + ix


def _masked_mean(
    values: np.ndarray,
    counts: np.ndarray,
    occupied: np.ndarray,
) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    result[occupied] = values[occupied] / counts[occupied]
    return result


def _nullable_list(values: np.ndarray) -> list[Optional[float]]:
    return [float(value) if math.isfinite(value) else None for value in values]


def main() -> int:
    args = _parse_args()
    payload = build_xy_statistics(
        args.inputs,
        args.output,
        bins=args.bins,
        max_particles=args.max_particles,
        chunk_size=args.chunk_size,
        position_scale=args.position_scale,
        extent=args.extent,
        verbose=True,
    )
    sampling = "all rows" if payload["sampling_is_exact"] else (
        f"1/{payload['sampling_stride']} systematic sample"
    )
    print(f"visualization = {payload['output']}")
    print(
        f"analyzed_rows = {payload['analyzed_count']} / "
        f"{payload['total_count']} ({sampling})"
    )
    return 0


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Particle x-y statistics</title>
<style>
:root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background: #08111f; color: #e7eef8; }
main { max-width: 1540px; margin: auto; padding: 28px; }
h1 { margin: 0 0 5px; font-size: 30px; }
h2 { margin: 0 0 4px; font-size: 17px; }
.subtle { color: #91a4bd; font-size: 13px; overflow-wrap: anywhere; }
.cards { display: grid; grid-template-columns: repeat(5, minmax(150px, 1fr)); gap: 12px; margin: 22px 0 14px; }
.card, section { background: #101c2f; border: 1px solid #243652; border-radius: 12px; box-shadow: 0 8px 26px #0004; }
.card { padding: 14px 16px; }
.label { color: #91a5c0; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; }
.value { margin-top: 5px; font-size: 20px; font-variant-numeric: tabular-nums; }
.note { margin: 0 0 14px; padding: 11px 14px; border-radius: 9px; background: #12253b; color: #b9c9dc; font-size: 13px; }
.plots { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
section { padding: 14px; min-width: 0; }
canvas { display: block; width: 100%; height: 350px; margin-top: 8px; cursor: crosshair; }
.plot-meta { color: #9badc6; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
.details { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 8px; }
th, td { border-bottom: 1px solid #263754; padding: 7px 5px; text-align: right; font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; }
.inputs { max-height: 310px; overflow: auto; }
.tooltip { position: fixed; display: none; z-index: 10; pointer-events: none; background: #050b14ee; border: 1px solid #48617f; border-radius: 7px; padding: 7px 9px; color: #dce8f7; font: 12px ui-monospace, monospace; white-space: pre-line; box-shadow: 0 5px 18px #0008; }
@media (max-width: 1050px) { .cards, .plots { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 720px) { main { padding: 16px; } .cards, .plots, .details { grid-template-columns: 1fr; } }
</style>
</head>
<body><main>
<h1 id="title"></h1>
<div class="subtle" id="subtitle"></div>
<div class="cards" id="cards"></div>
<div class="note" id="sampling-note"></div>
<div class="plots" id="plots"></div>
<div class="details">
  <section><h2>Analyzed-row summary</h2><div class="subtle">Positions use the normalized plot units; velocities retain source units.</div><div id="summary"></div></section>
  <section><h2>Input files</h2><div class="subtle">Rows from these files are aggregated into the maps above.</div><div class="inputs" id="inputs"></div></section>
</div>
<div class="tooltip" id="tooltip"></div>
</main>
<script>
const DATA = __EMBEDDED_DATA__;
const fmt = value => value == null ? "n/a" : Number(value).toLocaleString(undefined, {maximumSignificantDigits: 6});
const esc = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const PLOTS = [
  {key:"particle_count", title:"Particle density", note:"analyzed particles per bin", palette:"density", log:true},
  {key:"mean_z", title:"Mean z", note:"mean normalized z position", palette:"sequential"},
  {key:"mean_speed", title:"Mean speed", note:"mean sqrt(vx² + vy² + vz²)", palette:"sequential"},
  {key:"velocity_dispersion", title:"Velocity dispersion", note:"sqrt(E[|v|²] − |E[v]|²)", palette:"sequential"},
  {key:"mean_vx", title:"Mean vx", note:"diverging around zero", palette:"diverging"},
  {key:"mean_vy", title:"Mean vy", note:"diverging around zero", palette:"diverging"},
  {key:"mean_vz", title:"Mean vz", note:"diverging around zero", palette:"diverging"},
  {key:"velocity_vectors", title:"Mean planar velocity", note:"arrows show binned (mean vx, mean vy)", palette:"vectors"}
];
const layout = {left:58, right:18, top:16, bottom:46};
document.getElementById("title").textContent = DATA.title;
document.getElementById("subtitle").textContent = `${DATA.file_count} HDF5 file${DATA.file_count === 1 ? "" : "s"} · x-y extent [${fmt(DATA.extent.x_min)}, ${fmt(DATA.extent.x_max)}] × [${fmt(DATA.extent.y_min)}, ${fmt(DATA.extent.y_max)}]`;
const cards = [
  ["Input files", DATA.file_count], ["Particle rows", fmt(DATA.total_count)],
  ["Analyzed rows", fmt(DATA.analyzed_count)], ["Grid", `${DATA.bins} × ${DATA.bins}`],
  ["Occupied bins", fmt(DATA.grids.particle_count.filter(value => value > 0).length)]
];
document.getElementById("cards").innerHTML = cards.map(([label,value]) => `<div class="card"><div class="label">${label}</div><div class="value">${value}</div></div>`).join("");
document.getElementById("sampling-note").textContent = DATA.sampling_is_exact
  ? "Every source row was analyzed; maps and summaries are exact for the selected files."
  : `A deterministic global 1/${DATA.sampling_stride} systematic sample was analyzed. Counts are sample counts; binned means and summaries estimate the selected files.`;
document.getElementById("plots").innerHTML = PLOTS.map((plot, index) => `<section><h2>${plot.title}</h2><div class="plot-meta" id="meta-${index}">${plot.note}</div><canvas id="plot-${index}"></canvas></section>`).join("");

function extent(values, diverging=false, logarithmic=false) {
  let lo=Infinity, hi=-Infinity;
  for (const raw of values) { if (raw == null) continue; const value=logarithmic ? Math.log1p(raw) : raw; lo=Math.min(lo,value);hi=Math.max(hi,value); }
  if (!Number.isFinite(lo)) return [0,1];
  if (diverging) { const limit=Math.max(Math.abs(lo),Math.abs(hi),Number.EPSILON); return [-limit,limit]; }
  if (lo === hi) { const pad=Math.max(1,Math.abs(lo))*.05; return [lo-pad,hi+pad]; }
  return [lo,hi];
}
function interpolate(a,b,t) { return Math.round(a+(b-a)*t); }
function color(t,palette) {
  t=Math.max(0,Math.min(1,t));
  let stops;
  if (palette === "diverging") stops=[[42,93,166],[201,218,238],[247,247,247],[244,165,130],[178,24,43]];
  else if (palette === "density") stops=[[7,18,38],[30,75,124],[29,145,192],[94,201,98],[250,231,91]];
  else stops=[[15,32,63],[33,101,143],[38,166,164],[132,211,125],[247,226,93]];
  const scaled=t*(stops.length-1), index=Math.min(stops.length-2,Math.floor(scaled)), local=scaled-index;
  return [interpolate(stops[index][0],stops[index+1][0],local),interpolate(stops[index][1],stops[index+1][1],local),interpolate(stops[index][2],stops[index+1][2],local),255];
}
function setup(canvas) {
  const ratio=window.devicePixelRatio||1, rect=canvas.getBoundingClientRect();
  canvas.width=Math.max(1,Math.round(rect.width*ratio));canvas.height=Math.max(1,Math.round(rect.height*ratio));
  const ctx=canvas.getContext("2d");ctx.scale(ratio,ratio);return [ctx,rect.width,rect.height];
}
function axes(ctx,w,h) {
  const pw=w-layout.left-layout.right,ph=h-layout.top-layout.bottom;
  ctx.strokeStyle="#2a3c59";ctx.fillStyle="#9badc6";ctx.lineWidth=1;ctx.font="11px system-ui";
  for(let i=0;i<=4;i++){const px=layout.left+pw*i/4,py=layout.top+ph*i/4;ctx.beginPath();ctx.moveTo(px,layout.top);ctx.lineTo(px,layout.top+ph);ctx.stroke();ctx.beginPath();ctx.moveTo(layout.left,py);ctx.lineTo(layout.left+pw,py);ctx.stroke();}
  ctx.fillText(fmt(DATA.extent.x_min),layout.left,layout.top+ph+18);ctx.textAlign="right";ctx.fillText(fmt(DATA.extent.x_max),layout.left+pw,layout.top+ph+18);ctx.textAlign="left";
  ctx.fillText(fmt(DATA.extent.y_max),3,layout.top+4);ctx.fillText(fmt(DATA.extent.y_min),3,layout.top+ph);ctx.textAlign="center";ctx.fillText("x",layout.left+pw/2,h-5);
  ctx.save();ctx.translate(12,layout.top+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText("y",0,0);ctx.restore();ctx.textAlign="left";
  return {pw,ph};
}
function drawRaster(ctx,w,h,values,palette,logarithmic) {
  const domain=extent(values,palette==="diverging",logarithmic), bins=DATA.bins;
  const scratch=document.createElement("canvas");scratch.width=bins;scratch.height=bins;const sctx=scratch.getContext("2d"),image=sctx.createImageData(bins,bins);
  for(let iy=0;iy<bins;iy++)for(let ix=0;ix<bins;ix++){
    const value=values[iy*bins+ix],target=((bins-1-iy)*bins+ix)*4;
    if(value==null || (palette==="density" && value===0)){image.data[target]=12;image.data[target+1]=23;image.data[target+2]=39;image.data[target+3]=255;continue;}
    const transformed=logarithmic?Math.log1p(value):value,t=(transformed-domain[0])/(domain[1]-domain[0]),rgba=color(Number.isFinite(t)?t:.5,palette);
    for(let c=0;c<4;c++)image.data[target+c]=rgba[c];
  }
  sctx.putImageData(image,0,0);ctx.imageSmoothingEnabled=true;ctx.drawImage(scratch,layout.left,layout.top,w-layout.left-layout.right,h-layout.top-layout.bottom);
  return domain;
}
function drawVectors(ctx,w,h) {
  const bins=DATA.bins,counts=DATA.grids.particle_count,vx=DATA.grids.mean_vx,vy=DATA.grids.mean_vy;
  drawRaster(ctx,w,h,counts,"density",true);const pw=w-layout.left-layout.right,ph=h-layout.top-layout.bottom;
  const step=Math.max(1,Math.round(bins/18));let max=0;
  for(let iy=0;iy<bins;iy+=step)for(let ix=0;ix<bins;ix+=step){const i=iy*bins+ix;if(vx[i]!=null)max=Math.max(max,Math.hypot(vx[i],vy[i]));}
  if(max===0)return;const arrowLength=Math.min(pw,ph)/24;ctx.strokeStyle="#ffffffdd";ctx.fillStyle="#ffffffdd";ctx.lineWidth=1.2;
  for(let iy=Math.floor(step/2);iy<bins;iy+=step)for(let ix=Math.floor(step/2);ix<bins;ix+=step){const i=iy*bins+ix;if(vx[i]==null)continue;const magnitude=Math.hypot(vx[i],vy[i]);if(!magnitude)continue;const x=layout.left+(ix+.5)/bins*pw,y=layout.top+ph-(iy+.5)/bins*ph,dx=vx[i]/max*arrowLength,dy=-vy[i]/max*arrowLength,angle=Math.atan2(dy,dx),tx=x+dx,ty=y+dy;ctx.beginPath();ctx.moveTo(x-dx, y-dy);ctx.lineTo(tx,ty);ctx.stroke();ctx.beginPath();ctx.moveTo(tx,ty);ctx.lineTo(tx-5*Math.cos(angle-.55),ty-5*Math.sin(angle-.55));ctx.lineTo(tx-5*Math.cos(angle+.55),ty-5*Math.sin(angle+.55));ctx.closePath();ctx.fill();}
}
function drawPlot(plot,index) {
  const canvas=document.getElementById(`plot-${index}`),[ctx,w,h]=setup(canvas);ctx.fillStyle="#0c1727";ctx.fillRect(0,0,w,h);
  let domain;
  if(plot.palette==="vectors"){drawVectors(ctx,w,h);domain=null;}else{domain=drawRaster(ctx,w,h,DATA.grids[plot.key],plot.palette,plot.log);}
  axes(ctx,w,h);
  if(domain)document.getElementById(`meta-${index}`).textContent=`${plot.note} · range ${fmt(plot.log?Math.expm1(domain[0]):domain[0])} to ${fmt(plot.log?Math.expm1(domain[1]):domain[1])}`;
  canvas.onmousemove=event=>showTooltip(event,plot);canvas.onmouseleave=()=>document.getElementById("tooltip").style.display="none";
}
function showTooltip(event,plot) {
  const rect=event.currentTarget.getBoundingClientRect(),pw=rect.width-layout.left-layout.right,ph=rect.height-layout.top-layout.bottom,px=event.clientX-rect.left-layout.left,py=event.clientY-rect.top-layout.top;
  if(px<0||py<0||px>=pw||py>=ph)return;
  const ix=Math.min(DATA.bins-1,Math.floor(px/pw*DATA.bins)),iy=Math.min(DATA.bins-1,Math.floor((1-py/ph)*DATA.bins)),i=iy*DATA.bins+ix;
  const x0=DATA.extent.x_min+(DATA.extent.x_max-DATA.extent.x_min)*ix/DATA.bins,x1=DATA.extent.x_min+(DATA.extent.x_max-DATA.extent.x_min)*(ix+1)/DATA.bins,y0=DATA.extent.y_min+(DATA.extent.y_max-DATA.extent.y_min)*iy/DATA.bins,y1=DATA.extent.y_min+(DATA.extent.y_max-DATA.extent.y_min)*(iy+1)/DATA.bins;
  let value=plot.palette==="vectors"?`vx: ${fmt(DATA.grids.mean_vx[i])}\nvy: ${fmt(DATA.grids.mean_vy[i])}`:`value: ${fmt(DATA.grids[plot.key][i])}`;
  const tip=document.getElementById("tooltip");tip.textContent=`x: ${fmt(x0)} – ${fmt(x1)}\ny: ${fmt(y0)} – ${fmt(y1)}\ncount: ${fmt(DATA.grids.particle_count[i])}\n${value}`;tip.style.display="block";tip.style.left=`${Math.min(window.innerWidth-210,event.clientX+14)}px`;tip.style.top=`${Math.min(window.innerHeight-105,event.clientY+14)}px`;
}
document.getElementById("summary").innerHTML=`<table><thead><tr><th>field</th><th>minimum</th><th>maximum</th><th>mean</th><th>std. dev.</th></tr></thead><tbody>${Object.entries(DATA.summaries).map(([name,row])=>`<tr><td>${name}</td><td>${fmt(row.minimum)}</td><td>${fmt(row.maximum)}</td><td>${fmt(row.mean)}</td><td>${fmt(row.standard_deviation)}</td></tr>`).join("")}</tbody></table>`;
document.getElementById("inputs").innerHTML=`<table><thead><tr><th>file</th><th>rank</th><th>rows</th><th>mesh</th><th>scale</th></tr></thead><tbody>${DATA.inputs.map(row=>`<tr title="${esc(row.path)}"><td>${esc(row.name)}</td><td>${fmt(row.rank)}</td><td>${fmt(row.count)}</td><td>${fmt(row.nsidemesh)}</td><td>${fmt(row.position_scale)}</td></tr>`).join("")}</tbody></table>`;
function redraw(){PLOTS.forEach(drawPlot);} redraw();let resizeTimer;window.addEventListener("resize",()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(redraw,100);});
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
