"""Visualize why the periodic lattice layout improves compression ratio.

The dashboard compares a sort-only manifest with a lattice-layout manifest.
An optional payload-search JSON adds the causal flat -> dense -> residual
ablations produced by :mod:`experiments.payload_search`.  The result is one
self-contained HTML file with no plotting dependency.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional


FIELDS = ("x", "y", "z", "vx", "vy", "vz")
POSITION_FIELDS = ("x", "y", "z")
VELOCITY_FIELDS = ("vx", "vy", "vz")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "flat_report",
        type=Path,
        help="Sort-only work directory or manifest.json.",
    )
    parser.add_argument(
        "lattice_report",
        type=Path,
        help="Lattice-layout work directory or manifest.json.",
    )
    parser.add_argument(
        "--ablation-json",
        type=Path,
        help="Optional JSON written by experiments.payload_search.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lattice_layout_advantage.html"),
        help=(
            "Self-contained HTML output "
            "(default: lattice_layout_advantage.html)."
        ),
    )
    return parser.parse_args()


def build_lattice_advantage_visualization(
    flat_report: Path,
    lattice_report: Path,
    output: Path,
    *,
    ablation_json: Optional[Path] = None,
) -> dict[str, Any]:
    """Build the comparison dashboard and return its embedded payload."""

    flat_path, flat = _read_report(flat_report)
    lattice_path, lattice = _read_report(lattice_report)
    _validate_reports(flat, lattice)

    count = int(flat["count"])
    flat_sizes = _field_sizes(flat)
    lattice_sizes = _field_sizes(lattice)
    flat_total = int(flat["sizes"]["compressed_total_bytes"])
    lattice_total = int(lattice["sizes"]["compressed_total_bytes"])
    original_bytes = int(flat["sizes"]["selected_original_payload_bytes"])
    flat_cr = float(flat["sizes"]["payload_compression_ratio"])
    lattice_cr = float(lattice["sizes"]["payload_compression_ratio"])

    position_flat = sum(flat_sizes[field] for field in POSITION_FIELDS)
    position_lattice = sum(
        lattice_sizes[field] for field in POSITION_FIELDS
    )
    velocity_flat = sum(flat_sizes[field] for field in VELOCITY_FIELDS)
    velocity_lattice = sum(
        lattice_sizes[field] for field in VELOCITY_FIELDS
    )
    id_flat = _field_component_bytes(flat, "id")
    id_lattice = _field_component_bytes(lattice, "id")
    group_savings = {
        "positions": position_flat - position_lattice,
        "velocities": velocity_flat - velocity_lattice,
        "id": id_flat - id_lattice,
    }
    accounted = sum(group_savings.values())
    group_savings["metadata/other"] = (
        flat_total - lattice_total - accounted
    )

    lattice_metadata = lattice["lattice_layout"]
    dense_count = int(lattice_metadata["dense_count"])
    payload: dict[str, Any] = {
        "title": "Why the periodic lattice layout compresses better",
        "input_name": Path(str(flat.get("input_h5", "input"))).name,
        "flat_report": str(flat_path),
        "lattice_report": str(lattice_path),
        "codec": _lossy_codec(flat),
        "count": count,
        "original_bytes": original_bytes,
        "flat": {
            "compression_ratio": flat_cr,
            "compressed_bytes": flat_total,
            "field_bytes": flat_sizes,
        },
        "lattice": {
            "compression_ratio": lattice_cr,
            "compressed_bytes": lattice_total,
            "field_bytes": lattice_sizes,
            "shape": [int(value) for value in lattice_metadata["dense_shape"]],
            "occupancy": float(lattice_metadata["occupancy"]),
            "dense_count": dense_count,
            "dense_overhead_fraction": dense_count / count - 1.0,
            "axis_search": bool(lattice_metadata.get("axis_search", False)),
            "hole_fill": str(lattice_metadata.get("hole_fill", "unknown")),
        },
        "comparison": {
            "bytes_saved": flat_total - lattice_total,
            "compressed_size_reduction_fraction": 1.0
            - lattice_total / flat_total,
            "compression_ratio_increase_fraction": lattice_cr / flat_cr - 1.0,
            "position_size_reduction_fraction": 1.0
            - position_lattice / position_flat,
            "velocity_size_reduction_fraction": 1.0
            - velocity_lattice / velocity_flat,
            "group_savings": group_savings,
        },
        "ablation": None,
    }
    if ablation_json is not None:
        payload["ablation"] = _read_ablation(
            ablation_json,
            count,
            lattice_sizes,
        )

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


def _read_report(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "manifest.json"
    if not resolved.is_file():
        raise RuntimeError(f"Report does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Report is not a JSON object: {resolved}")
    return resolved, payload


def _lossy_codec(report: Mapping[str, Any]) -> str:
    compressors = report.get("compressors", {})
    return str(
        compressors.get(
            "lossy",
            compressors.get("positions", compressors.get("velocities", "unknown")),
        )
    )


def _bound(report: Mapping[str, Any], field: str) -> float:
    entry = report.get("field_error_bounds", {}).get(field, {})
    value = entry.get("compressor_abs", entry.get("abs"))
    if value is None:
        raise RuntimeError(f"Report has no error bound for {field!r}.")
    return float(value)


def _validate_reports(
    flat: Mapping[str, Any],
    lattice: Mapping[str, Any],
) -> None:
    if int(flat.get("count", -1)) != int(lattice.get("count", -2)):
        raise RuntimeError("Flat and lattice reports have different row counts.")
    if _lossy_codec(flat) != _lossy_codec(lattice):
        raise RuntimeError("Flat and lattice reports use different lossy codecs.")
    flat_layout = flat.get("lattice_layout") or {}
    lattice_layout = lattice.get("lattice_layout") or {}
    if bool(flat_layout.get("enabled", False)):
        raise RuntimeError("The flat report already has lattice layout enabled.")
    if not bool(lattice_layout.get("enabled", False)):
        raise RuntimeError("The lattice report does not have lattice layout enabled.")
    for report, label in ((flat, "flat"), (lattice, "lattice")):
        ordering = report.get("particle_sort", {})
        if not bool(ordering.get("enabled", False)):
            raise RuntimeError(f"The {label} report is not ID-sorted.")
    flat_original = int(flat["sizes"]["selected_original_payload_bytes"])
    lattice_original = int(
        lattice["sizes"]["selected_original_payload_bytes"]
    )
    if flat_original != lattice_original:
        raise RuntimeError("Reports use different original payload sizes.")
    for field in FIELDS:
        if not math.isclose(
            _bound(flat, field),
            _bound(lattice, field),
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise RuntimeError(
                f"Flat and lattice reports use different {field} bounds."
            )


def _field_component_bytes(report: Mapping[str, Any], field: str) -> int:
    components = report.get("sizes", {}).get(
        "compressed_components_bytes",
        {},
    )
    prefix = f"compressed/{field}."
    return sum(
        int(size)
        for path, size in components.items()
        if str(path).startswith(prefix)
    )


def _field_sizes(report: Mapping[str, Any]) -> dict[str, int]:
    return {
        field: _field_component_bytes(report, field)
        for field in FIELDS
    }


def _read_ablation(
    path: Path,
    expected_count: int,
    production_sizes: Mapping[str, int],
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"Ablation JSON does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if int(payload.get("count", -1)) != expected_count:
        raise RuntimeError(
            "Ablation JSON and manifests have different row counts."
        )
    indexed = {
        (str(row["field"]), str(row["layout"])): int(row["compressed_bytes"])
        for row in payload.get("results", [])
        if isinstance(row, dict)
        and "field" in row
        and "layout" in row
        and "compressed_bytes" in row
    }

    rows: dict[str, list[dict[str, Any]]] = {}
    for field in FIELDS:
        stages = []
        for label, layout in (
            ("ID-sorted 1-D", "id_sorted_1d:auto"),
            ("dense 3-D", "dense_3d:auto"),
        ):
            value = indexed.get((field, layout))
            if value is not None:
                stages.append({"stage": label, "compressed_bytes": value})
        if field in POSITION_FIELDS:
            value = indexed.get(
                (field, "lattice_residual_dense_3d:auto")
            )
            if value is not None:
                stages.append(
                    {
                        "stage": "3-D + lattice residual",
                        "compressed_bytes": value,
                    }
                )
        stages.append(
            {
                "stage": "production lattice",
                "compressed_bytes": int(production_sizes[field]),
            }
        )
        rows[field] = stages
    return {"source": str(resolved), "fields": rows}


def main() -> int:
    args = _parse_args()
    payload = build_lattice_advantage_visualization(
        args.flat_report,
        args.lattice_report,
        args.output,
        ablation_json=args.ablation_json,
    )
    comparison = payload["comparison"]
    print(f"visualization = {payload['output']}")
    print(
        "payload_CR = "
        f"{payload['flat']['compression_ratio']:.6g} (sort only) -> "
        f"{payload['lattice']['compression_ratio']:.6g} (lattice)"
    )
    print(
        "compressed_bytes_saved = "
        f"{comparison['bytes_saved']} "
        f"({comparison['compressed_size_reduction_fraction']:.2%})"
    )
    return 0


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lattice-layout compression advantage</title>
<style>
:root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background: #08111f; color: #eaf0fa; }
main { max-width: 1400px; margin: auto; padding: 28px; }
h1 { margin: 0 0 6px; font-size: 30px; }
h2 { margin: 0 0 5px; font-size: 17px; }
.subtle { color: #93a6c1; font-size: 13px; }
.cards { display: grid; grid-template-columns: repeat(5, minmax(150px, 1fr)); gap: 12px; margin: 22px 0 14px; }
.card, section { background: #101d30; border: 1px solid #263a59; border-radius: 12px; box-shadow: 0 8px 28px #0004; }
.card { padding: 14px 16px; }
.label { color: #91a7c6; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; }
.value { margin-top: 5px; font-size: 21px; font-variant-numeric: tabular-nums; }
.gain { color: #61e4a8; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 14px; }
section { padding: 15px; min-width: 0; }
canvas { display: block; width: 100%; height: 330px; margin-top: 9px; }
.wide { grid-column: 1 / -1; }
.mechanism { display: grid; grid-template-columns: 1fr 70px 1fr; align-items: center; gap: 14px; }
.mode { border: 1px solid #2d4568; border-radius: 10px; padding: 14px; min-height: 170px; }
.mode.flat { background: #141d2d; } .mode.lattice { background: #10293a; }
.arrow { text-align: center; font-size: 36px; color: #61e4a8; }
.chain { display: flex; align-items: center; gap: 3px; margin: 24px 0; overflow: hidden; }
.dot { width: 11px; height: 11px; border-radius: 50%; flex: none; background: #f2a65a; }
.link { height: 2px; min-width: 13px; flex: 1; background: #526984; }
.cube { position: relative; height: 74px; margin: 14px 0 10px; }
.cube .axis { position: absolute; left: 50%; top: 50%; height: 2px; width: 40%; transform-origin: left; background: #61e4a8; }
.cube .a0 { transform: rotate(-90deg); }.cube .a1 { transform: rotate(-25deg); }.cube .a2 { transform: rotate(25deg); }
.cube .center { position: absolute; left: calc(50% - 7px); top: calc(50% - 7px); width: 14px; height: 14px; border-radius: 50%; background: #f2a65a; box-shadow: 0 0 18px #f2a65aaa; }
.points { font-size: 13px; line-height: 1.55; color: #b4c4da; }
.points strong { color: #eaf0fa; }
.foot { margin-top: 13px; overflow-wrap: anywhere; }
@media (max-width: 1000px) { .cards { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 760px) { main { padding: 16px; } .cards, .grid { grid-template-columns: 1fr; } .wide { grid-column: auto; } .mechanism { grid-template-columns: 1fr; } .arrow { transform: rotate(90deg); } }
</style>
</head>
<body><main>
<h1 id="title"></h1>
<div class="subtle" id="subtitle"></div>
<div class="cards" id="cards"></div>
<div class="grid">
  <section><h2>Position streams</h2><div class="subtle">Actual compressed bytes, including lattice wrap sidecars.</div><canvas id="positions"></canvas></section>
  <section><h2>Velocity streams</h2><div class="subtle">Most of the package-level saving comes from restored 3-D neighborhoods.</div><canvas id="velocities"></canvas></section>
  <section><h2>Where the saved bytes come from</h2><div class="subtle">Positive bars save space; a negative bar is extra lattice metadata.</div><canvas id="savings"></canvas></section>
  <section><h2>Causal layout ablation</h2><div class="subtle" id="ablation-note">Run experiments.payload_search to add flat → dense → residual results.</div><canvas id="ablation"></canvas></section>
  <section class="wide">
    <div class="mechanism">
      <div class="mode flat"><h2>ID-sorted flat stream</h2><div class="chain" id="chain"></div><div class="points">One numeric walk through ID space. The codec sees a <strong>1-D shape</strong>, so row/plane neighbors and periodic joins are not represented as neighbors in the prediction geometry.</div></div>
      <div class="arrow">→</div>
      <div class="mode lattice"><h2>Periodic dense lattice</h2><div class="cube"><span class="axis a0"></span><span class="axis a1"></span><span class="axis a2"></span><span class="center"></span></div><div class="points">The three ID digits become <strong>three codec axes</strong>. Periodic rank seams are unwrapped, sparse holes are prediction-only fills, and positions are coded as residuals from their known lattice coordinate.</div></div>
    </div>
  </section>
</div>
<div class="subtle foot" id="provenance"></div>
</main>
<script>
const DATA = __EMBEDDED_DATA__;
const COLORS = {flat:"#f2a65a", lattice:"#55c7f3", residual:"#a78bfa", production:"#61e4a8", grid:"#2a3d5c", text:"#a9bad0", negative:"#ff7288"};
const fmt = value => Number(value).toLocaleString(undefined, {maximumSignificantDigits: 5});
const bytes = value => value >= 1e6 ? `${(value/1e6).toFixed(2)} MB` : value >= 1e3 ? `${(value/1e3).toFixed(1)} kB` : `${value} B`;
const pct = value => `${(100*value).toFixed(2)}%`;
document.getElementById("title").textContent = DATA.title;
document.getElementById("subtitle").textContent = `${DATA.input_name} · ${fmt(DATA.count)} particles · ${DATA.codec.toUpperCase()} · identical field bounds`;
const cards = [
  ["Sort-only CR", fmt(DATA.flat.compression_ratio), ""],
  ["Lattice CR", fmt(DATA.lattice.compression_ratio), "gain"],
  ["CR increase", `+${pct(DATA.comparison.compression_ratio_increase_fraction)}`, "gain"],
  ["Payload bytes saved", bytes(DATA.comparison.bytes_saved), "gain"],
  ["Dense occupancy", pct(DATA.lattice.occupancy), ""]
];
document.getElementById("cards").innerHTML = cards.map(([label,value,klass]) => `<div class="card"><div class="label">${label}</div><div class="value ${klass}">${value}</div></div>`).join("");
document.getElementById("chain").innerHTML = Array.from({length:10}, (_,i) => `${i?'<span class="link"></span>':''}<span class="dot"></span>`).join("");
document.getElementById("provenance").textContent = `Reports: ${DATA.flat_report} and ${DATA.lattice_report}. Dense shape ${DATA.lattice.shape.join(" × ")}; encoded-cell overhead ${pct(DATA.lattice.dense_overhead_fraction)}; hole fill ${DATA.lattice.hole_fill}; axis search ${DATA.lattice.axis_search ? "enabled" : "disabled"}.`;

function setup(canvas) {
  const ratio=window.devicePixelRatio||1, rect=canvas.getBoundingClientRect();
  canvas.width=Math.max(1,Math.round(rect.width*ratio)); canvas.height=Math.max(1,Math.round(rect.height*ratio));
  const ctx=canvas.getContext("2d"); ctx.scale(ratio,ratio); return [ctx,rect.width,rect.height];
}
function grouped(canvas, fields, series) {
  const [ctx,w,h]=setup(canvas), m={l:64,r:12,t:16,b:58}, pw=w-m.l-m.r, ph=h-m.t-m.b;
  const max=Math.max(...series.flatMap(s=>fields.map(f=>s.values[f])),1)*1.12;
  ctx.font="11px system-ui"; ctx.fillStyle=COLORS.text; ctx.strokeStyle=COLORS.grid;
  for(let i=0;i<=4;i++){const y=m.t+ph*i/4;ctx.beginPath();ctx.moveTo(m.l,y);ctx.lineTo(m.l+pw,y);ctx.stroke();ctx.fillText(bytes(max*(1-i/4)),3,y+4);}
  const groupW=pw/fields.length, barW=Math.min(28,groupW/(series.length+1));
  fields.forEach((field,fi)=>{const cx=m.l+groupW*(fi+.5);ctx.textAlign="center";ctx.fillStyle="#dbe7f6";ctx.fillText(field,cx,m.t+ph+18);series.forEach((s,si)=>{const v=s.values[field], bh=ph*v/max, x=cx+(si-(series.length-1)/2)*barW-barW*.42;ctx.fillStyle=s.color;ctx.fillRect(x,m.t+ph-bh,barW*.84,bh);});});
  ctx.textAlign="left"; let lx=m.l; series.forEach(s=>{ctx.fillStyle=s.color;ctx.fillRect(lx,m.t+ph+35,10,10);ctx.fillStyle=COLORS.text;ctx.fillText(s.name,lx+14,m.t+ph+44);lx+=ctx.measureText(s.name).width+35;});
}
function savings(canvas) {
  const [ctx,w,h]=setup(canvas), rows=Object.entries(DATA.comparison.group_savings), m={l:105,r:65,t:20,b:20}, pw=w-m.l-m.r, rowH=(h-m.t-m.b)/rows.length;
  const max=Math.max(...rows.map(([,v])=>Math.abs(v)),1), zero=m.l+pw*.18, posW=pw*.82;
  ctx.font="12px system-ui"; rows.forEach(([name,value],i)=>{const y=m.t+i*rowH+rowH*.25, width=Math.abs(value)/max*posW;ctx.fillStyle=COLORS.text;ctx.textAlign="right";ctx.fillText(name,m.l-9,y+13);ctx.fillStyle=value>=0?COLORS.production:COLORS.negative;ctx.fillRect(value>=0?zero:zero-width,y,width,rowH*.45);ctx.fillStyle="#dbe7f6";ctx.textAlign=value>=0?"left":"right";ctx.fillText(`${value>=0?"+":"−"}${bytes(Math.abs(value))}`,value>=0?zero+width+6:zero-width-6,y+13);});
  ctx.strokeStyle=COLORS.grid;ctx.beginPath();ctx.moveTo(zero,m.t);ctx.lineTo(zero,h-m.b);ctx.stroke();ctx.textAlign="left";
}
function ablation(canvas) {
  const [ctx,w,h]=setup(canvas); if(!DATA.ablation){ctx.fillStyle=COLORS.text;ctx.font="14px system-ui";ctx.fillText("No ablation JSON supplied.",20,40);return;}
  document.getElementById("ablation-note").textContent="Aggregate compressed bytes; positions show the extra residual step, velocities isolate 3-D reshaping.";
  const groups=[
    ["positions",["x","y","z"]], ["velocities",["vx","vy","vz"]]
  ].map(([name,fields])=>({name,stages:aggregateStages(fields)}));
  const m={l:72,r:15,t:20,b:55},pw=w-m.l-m.r,ph=h-m.t-m.b,max=Math.max(...groups.flatMap(g=>g.stages.map(s=>s.value)),1)*1.12;
  ctx.font="11px system-ui";ctx.strokeStyle=COLORS.grid;ctx.fillStyle=COLORS.text;for(let i=0;i<=4;i++){const y=m.t+ph*i/4;ctx.beginPath();ctx.moveTo(m.l,y);ctx.lineTo(m.l+pw,y);ctx.stroke();ctx.fillText(bytes(max*(1-i/4)),2,y+4);}
  const colors=[COLORS.flat,COLORS.lattice,COLORS.residual,COLORS.production],groupW=pw/groups.length;
  groups.forEach((g,gi)=>{const barW=Math.min(34,groupW/(g.stages.length+1)),cx=m.l+groupW*(gi+.5);g.stages.forEach((s,si)=>{const bh=ph*s.value/max,x=cx+(si-(g.stages.length-1)/2)*barW-barW*.4;ctx.fillStyle=colors[si];ctx.fillRect(x,m.t+ph-bh,barW*.8,bh);ctx.save();ctx.translate(x+barW*.4,m.t+ph+7);ctx.rotate(-.55);ctx.fillStyle=COLORS.text;ctx.textAlign="right";ctx.fillText(s.stage,0,0);ctx.restore();});ctx.textAlign="center";ctx.fillStyle="#dbe7f6";ctx.fillText(g.name,cx,h-3);});ctx.textAlign="left";
}
function aggregateStages(fields){const byName=new Map();for(const field of fields){for(const row of DATA.ablation.fields[field])byName.set(row.stage,(byName.get(row.stage)||0)+row.compressed_bytes);}return Array.from(byName,([stage,value])=>({stage,value}));}
function draw(){grouped(document.getElementById("positions"),["x","y","z"],[{name:"ID-sorted 1-D",color:COLORS.flat,values:DATA.flat.field_bytes},{name:"production lattice",color:COLORS.production,values:DATA.lattice.field_bytes}]);grouped(document.getElementById("velocities"),["vx","vy","vz"],[{name:"ID-sorted 1-D",color:COLORS.flat,values:DATA.flat.field_bytes},{name:"production lattice",color:COLORS.production,values:DATA.lattice.field_bytes}]);savings(document.getElementById("savings"));ablation(document.getElementById("ablation"));}
draw();window.addEventListener("resize",draw);
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
