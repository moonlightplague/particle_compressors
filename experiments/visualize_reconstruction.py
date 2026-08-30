"""Create a self-contained reconstruction-quality HTML dashboard.

The dashboard compares a completed roundtrip's original and reconstructed
particle fields. It honors the package's recorded row ordering, samples rows
deterministically, and embeds all plot data in one portable HTML file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import LOGICAL_ORDER, POSITION_FIELDS  # noqa: E402
from src.metrics import comparison_order_for_reconstructed_rows  # noqa: E402
from src.runtime import read_json  # noqa: E402


LOSSY_FIELDS = tuple(field for field in LOGICAL_ORDER if field != "id")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "work_dir",
        type=Path,
        help="Completed roundtrip package containing manifest.json.",
    )
    parser.add_argument(
        "--original-h5",
        type=Path,
        help="Override the original HDF5 path recorded in the manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output HTML path (default: WORK_DIR/"
            "reconstruction_visualization.html)."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5_000,
        help="Maximum particle rows embedded in the dashboard.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for deterministic row sampling.",
    )
    return parser.parse_args()


def build_visualization(
    work_dir: Path,
    output: Optional[Path] = None,
    original_h5: Optional[Path] = None,
    sample_size: int = 5_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a reconstruction dashboard and return its embedded data."""
    work_dir = work_dir.expanduser().resolve()
    manifest_path = work_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Manifest does not exist: {manifest_path}")
    if sample_size <= 0:
        raise RuntimeError("--sample-size must be positive.")

    manifest = read_json(manifest_path)
    source_path = _resolve_original_path(manifest, original_h5)
    reconstructed_path = _resolve_reconstructed_path(work_dir, manifest)
    metrics_path = work_dir / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.is_file() else {}
    count = int(manifest["count"])
    if count <= 0:
        raise RuntimeError("Cannot visualize a package with no particle rows.")

    sample_indices = _sample_indices(count, sample_size, seed)
    fields: dict[str, dict[str, Any]] = {}
    sampled_ids_exact = False
    scale = float(manifest["position_scale"]["value"])
    with h5py.File(source_path, "r") as source, h5py.File(
        reconstructed_path,
        "r",
    ) as reconstructed:
        comparison_order, alignment_source = (
            comparison_order_for_reconstructed_rows(
                source,
                reconstructed,
                manifest,
                count,
            )
        )
        original_indices = (
            sample_indices
            if comparison_order is None
            else comparison_order[sample_indices]
        )
        for logical in LOGICAL_ORDER:
            dataset_path = str(manifest["fields"][logical]["h5_path"])
            original_values = _read_selected(
                source[dataset_path],
                original_indices,
            )
            reconstructed_values = _read_selected(
                reconstructed[dataset_path],
                sample_indices,
            )
            if logical == "id":
                sampled_ids_exact = bool(
                    np.array_equal(original_values, reconstructed_values)
                )
                continue
            if logical in POSITION_FIELDS:
                original_values = original_values.astype(np.float64) / scale
                reconstructed_values = (
                    reconstructed_values.astype(np.float64) / scale
                )
            else:
                original_values = original_values.astype(np.float64)
                reconstructed_values = reconstructed_values.astype(np.float64)
            difference = reconstructed_values - original_values
            fields[logical] = _field_payload(
                logical,
                original_values,
                reconstructed_values,
                difference,
                manifest,
                metrics,
            )

    payload = {
        "title": f"Reconstruction: {source_path.name}",
        "original_h5": str(source_path),
        "reconstructed_h5": str(reconstructed_path),
        "work_dir": str(work_dir),
        "codec": manifest.get("compressors", {}).get("lossy", "unknown"),
        "count": count,
        "sample_count": int(sample_indices.size),
        "sample_seed": int(seed),
        "alignment_source": alignment_source,
        "payload_compression_ratio": (
            metrics.get("sizes", {}).get("payload_compression_ratio")
        ),
        "id_exact": bool(
            metrics.get("fields", {})
            .get("id", {})
            .get("exact_match", sampled_ids_exact)
        ),
        "id_exact_scope": (
            "full"
            if "exact_match" in metrics.get("fields", {}).get("id", {})
            else "sample"
        ),
        "lossy_fields": list(LOSSY_FIELDS),
        "fields": fields,
    }
    output_path = (
        output.expanduser().resolve()
        if output is not None
        else work_dir / "reconstruction_visualization.html"
    )
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


def _resolve_original_path(
    manifest: Mapping[str, Any],
    override: Optional[Path],
) -> Path:
    path = (
        override.expanduser().resolve()
        if override is not None
        else Path(str(manifest["input_h5"])).expanduser().resolve()
    )
    if not path.is_file():
        hint = " Pass --original-h5 PATH." if override is None else ""
        raise RuntimeError(f"Original HDF5 file does not exist: {path}.{hint}")
    return path


def _resolve_reconstructed_path(
    work_dir: Path,
    manifest: Mapping[str, Any],
) -> Path:
    recorded = manifest.get("artifacts", {}).get("reconstructed_h5")
    path = Path(recorded).expanduser().resolve() if recorded else (
        work_dir / "reconstructed.h5"
    )
    if not path.is_file():
        raise RuntimeError(f"Reconstructed HDF5 file does not exist: {path}")
    return path


def _sample_indices(count: int, sample_size: int, seed: int) -> np.ndarray:
    if sample_size >= count:
        return np.arange(count, dtype=np.intp)
    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(count, size=sample_size, replace=False).astype(
            np.intp,
            copy=False,
        )
    )


def _read_selected(
    dataset: h5py.Dataset,
    indices: np.ndarray,
) -> np.ndarray:
    """Read arbitrary unique HDF5 rows while preserving requested order."""
    order = np.argsort(indices, kind="stable")
    sorted_values = np.asarray(dataset[indices[order]])
    values = np.empty_like(sorted_values)
    values[order] = sorted_values
    return values


def _field_payload(
    logical: str,
    original: np.ndarray,
    reconstructed: np.ndarray,
    difference: np.ndarray,
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    field_metrics = metrics.get("fields", {}).get(logical, {})
    consistency = metrics.get("error_bound_consistency", {}).get(logical, {})
    default_bound = manifest.get("field_error_bounds", {}).get(
        logical,
        {},
    ).get("abs")
    bound = consistency.get("effective_final_abs_bound", default_bound)
    sample_max = float(np.abs(difference).max(initial=0.0))
    sample_rmse = math.sqrt(float(np.mean(np.square(difference))))
    return {
        "name": logical,
        "units": (
            "compressor_units" if logical in POSITION_FIELDS else "source_units"
        ),
        "original": original.tolist(),
        "reconstructed": reconstructed.tolist(),
        "error": difference.tolist(),
        "bound": None if bound is None else float(bound),
        "bound_satisfied": consistency.get("satisfied"),
        "max_absolute_error": float(
            field_metrics.get("max_absolute_error", sample_max)
        ),
        "rmse": float(field_metrics.get("rmse", sample_rmse)),
        "sample_max_absolute_error": sample_max,
    }


def main() -> int:
    args = _parse_args()
    payload = build_visualization(
        args.work_dir,
        output=args.output,
        original_h5=args.original_h5,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(f"visualization = {payload['output']}")
    print(
        f"sampled_rows = {payload['sample_count']} / {payload['count']} "
        f"({payload['alignment_source']})"
    )
    return 0


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Particle reconstruction visualization</title>
<style>
:root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
body { margin: 0; background: #09111f; color: #e8eef8; }
main { max-width: 1500px; margin: auto; padding: 28px; }
h1 { margin: 0 0 5px; font-size: 30px; }
.subtle { color: #8fa2bd; font-size: 13px; overflow-wrap: anywhere; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 22px 0; }
.card, section { background: #101c2f; border: 1px solid #243552; border-radius: 12px; box-shadow: 0 8px 26px #0004; }
.card { padding: 14px 16px; }
.label { color: #91a5c0; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; }
.value { margin-top: 5px; font-size: 21px; font-variant-numeric: tabular-nums; }
.overview { display: grid; grid-template-columns: 1.25fr .75fr; gap: 14px; }
section { padding: 14px; min-width: 0; }
h2 { margin: 0 0 4px; font-size: 17px; }
.plot-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-top: 14px; }
canvas { display: block; width: 100%; height: 310px; margin-top: 8px; }
.field-head { display: flex; justify-content: space-between; gap: 10px; align-items: baseline; }
.pass { color: #5de0a3; } .fail { color: #ff718b; } .unknown { color: #a7b5c9; }
.metrics { color: #9badc6; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
table { border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 8px; }
th, td { border-bottom: 1px solid #263754; padding: 7px 5px; text-align: right; font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; }
@media (max-width: 980px) { .cards, .plot-grid { grid-template-columns: repeat(2, 1fr); } .overview { grid-template-columns: 1fr; } }
@media (max-width: 620px) { main { padding: 16px; } .cards, .plot-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body><main>
<h1 id="title"></h1>
<div class="subtle" id="paths"></div>
<div class="cards" id="cards"></div>
<div class="overview">
  <section><h2>Spatial reconstruction error</h2><div class="subtle">Original x-y positions, colored blue-to-red from low to high sampled 3-D position error.</div><canvas id="spatial"></canvas></section>
  <section><h2>Field summary</h2><div class="subtle">Full-roundtrip metrics when metrics.json is available.</div><div id="summary"></div></section>
</div>
<div class="plot-grid" id="plots"></div>
</main>
<script>
const DATA = __EMBEDDED_DATA__;
const LOSSY = DATA.lossy_fields;
const COLORS = {grid:"#2a3c59", text:"#9badc6", point:"#62b7ff", bound:"#ff718b", zero:"#7890ae"};
const fmt = value => value == null ? "n/a" : Number(value).toLocaleString(undefined, {maximumSignificantDigits: 6});
document.getElementById("title").textContent = DATA.title;
document.getElementById("paths").textContent = `${DATA.original_h5} → ${DATA.reconstructed_h5}`;
const cards = [
  ["Lossy codec", DATA.codec], ["Particle rows", fmt(DATA.count)],
  ["Sampled rows", fmt(DATA.sample_count)], ["Payload CR", fmt(DATA.payload_compression_ratio)],
  ["ID match", DATA.id_exact ? (DATA.id_exact_scope === "full" ? "exact" : "sample exact") : "mismatch"]
];
document.getElementById("cards").innerHTML = cards.map(([label, value]) => `<div class="card"><div class="label">${label}</div><div class="value">${value}</div></div>`).join("");

function setup(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
  return [ctx, rect.width, rect.height];
}
function extent(values) {
  let lo = Infinity, hi = -Infinity;
  for (const value of values) { if (Number.isFinite(value)) { lo = Math.min(lo, value); hi = Math.max(hi, value); } }
  if (!Number.isFinite(lo)) return [-1, 1];
  if (lo === hi) { const pad = Math.max(1, Math.abs(lo)) * .05; return [lo-pad, hi+pad]; }
  const pad = (hi-lo) * .04; return [lo-pad, hi+pad];
}
function axes(ctx, width, height, xDomain, yDomain, xLabel, yLabel) {
  const m = {l:58, r:16, t:15, b:40}; const pw=width-m.l-m.r, ph=height-m.t-m.b;
  const sx = x => m.l + (x-xDomain[0])/(xDomain[1]-xDomain[0])*pw;
  const sy = y => m.t + (yDomain[1]-y)/(yDomain[1]-yDomain[0])*ph;
  ctx.strokeStyle=COLORS.grid; ctx.lineWidth=1; ctx.fillStyle=COLORS.text; ctx.font="11px system-ui";
  for (let i=0; i<=4; i++) { const x=m.l+pw*i/4, y=m.t+ph*i/4; ctx.beginPath(); ctx.moveTo(x,m.t);ctx.lineTo(x,m.t+ph);ctx.stroke(); ctx.beginPath();ctx.moveTo(m.l,y);ctx.lineTo(m.l+pw,y);ctx.stroke(); }
  ctx.fillText(fmt(xDomain[0]),m.l,m.t+ph+18); ctx.textAlign="right";ctx.fillText(fmt(xDomain[1]),m.l+pw,m.t+ph+18);ctx.textAlign="left";
  ctx.fillText(fmt(yDomain[1]),4,m.t+4);ctx.fillText(fmt(yDomain[0]),4,m.t+ph); ctx.textAlign="center";ctx.fillText(xLabel,m.l+pw/2,height-4);
  ctx.save();ctx.translate(12,m.t+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText(yLabel,0,0);ctx.restore();ctx.textAlign="left";
  return {sx,sy,m,pw,ph};
}
function drawError(canvas, field) {
  const [ctx,w,h]=setup(canvas), x=field.original, y=field.error; const xd=extent(x);
  const sampled=Math.max(...y.map(Math.abs),0), limit=Math.max(sampled, field.bound || 0, Number.EPSILON)*1.12;
  const a=axes(ctx,w,h,xd,[-limit,limit],`original (${field.units})`,`reconstructed − original`);
  ctx.strokeStyle=COLORS.zero;ctx.beginPath();ctx.moveTo(a.m.l,a.sy(0));ctx.lineTo(a.m.l+a.pw,a.sy(0));ctx.stroke();
  if (field.bound != null) { ctx.strokeStyle=COLORS.bound;ctx.setLineDash([5,4]); for (const b of [-field.bound,field.bound]) {ctx.beginPath();ctx.moveTo(a.m.l,a.sy(b));ctx.lineTo(a.m.l+a.pw,a.sy(b));ctx.stroke();} ctx.setLineDash([]); }
  ctx.fillStyle=COLORS.point+"99"; for (let i=0;i<x.length;i++){ctx.beginPath();ctx.arc(a.sx(x[i]),a.sy(y[i]),1.7,0,Math.PI*2);ctx.fill();}
}
function drawSpatial() {
  const [ctx,w,h]=setup(document.getElementById("spatial")); const x=DATA.fields.x.original, y=DATA.fields.y.original;
  const dx=DATA.fields.x.error,dy=DATA.fields.y.error,dz=DATA.fields.z.error; const mag=dx.map((_,i)=>Math.hypot(dx[i],dy[i],dz[i]));
  const a=axes(ctx,w,h,extent(x),extent(y),"x (compressor units)","y (compressor units)"); const max=Math.max(...mag,Number.EPSILON);
  for(let i=0;i<x.length;i++){const t=Math.log1p(99*mag[i]/max)/Math.log(100);ctx.fillStyle=`rgb(${Math.round(65+190*t)},${Math.round(184-95*t)},${Math.round(255-110*t)})`;ctx.beginPath();ctx.arc(a.sx(x[i]),a.sy(y[i]),2,0,Math.PI*2);ctx.fill();}
}
function statusClass(value){return value===true?"pass":value===false?"fail":"unknown";}
document.getElementById("summary").innerHTML=`<table><thead><tr><th>field</th><th>max |error|</th><th>RMSE</th><th>bound</th></tr></thead><tbody>${LOSSY.map(name=>{const f=DATA.fields[name];return `<tr><td class="${statusClass(f.bound_satisfied)}">${name}</td><td>${fmt(f.max_absolute_error)}</td><td>${fmt(f.rmse)}</td><td>${fmt(f.bound)}</td></tr>`;}).join("")}</tbody></table><div class="subtle" style="margin-top:10px">Alignment: ${DATA.alignment_source}; seed: ${DATA.sample_seed}</div>`;
document.getElementById("plots").innerHTML=LOSSY.map(name=>{const f=DATA.fields[name];return `<section><div class="field-head"><h2>${name}</h2><span class="${statusClass(f.bound_satisfied)}">${f.bound_satisfied===true?"bound satisfied":f.bound_satisfied===false?"bound violated":"sample only"}</span></div><div class="metrics">sample max |error| = ${fmt(f.sample_max_absolute_error)}</div><canvas id="plot-${name}"></canvas></section>`;}).join("");
drawSpatial(); for(const name of LOSSY) drawError(document.getElementById(`plot-${name}`),DATA.fields[name]);
window.addEventListener("resize",()=>{drawSpatial();for(const name of LOSSY)drawError(document.getElementById(`plot-${name}`),DATA.fields[name]);});
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
