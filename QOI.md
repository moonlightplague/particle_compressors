# QoI frontend user manual

A **quantity of interest (QoI)** is a value calculated from the particle data,
such as speed or kinetic energy. A small error in each velocity component can
produce a larger error in that calculated value. This frontend lets you specify
the error you will accept in the QoI itself.

The initial implementation supports **XnYZip positions + SZO velocities**, the
combination already selected in `config.yaml`. It preserves one chosen QoI for
**every particle**, in addition to the existing velocity component bounds.
QoI processing is optional and disabled in the default configuration.

## Quick start with HACC

Run from the repository root, using the environment installed according to
[INSTALL.md](INSTALL.md). No additional packages or native builds are needed.

The directory present in this workspace is
`data/EXASKY-HACC-data-medium-size/`. It contains `xx.f32`, `yy.f32`, `zz.f32`,
`vx.f32`, `vy.f32`, and `vz.f32`. Pass the directory itself as the input.

Start with the first million particles and preserve speed:

```bash
python main.py roundtrip data/EXASKY-HACC-data-medium-size/ \
  --config config.yaml \
  --work-dir particle_pipeline_runs/hacc_qoi_speed \
  --qoi speed --qoi-rel-eb 1e-4 \
  --rel-eb 1e-3 --xnyzip-tie-sort \
  --limit 1000000 --metrics --clean-raw
```

Here `--rel-eb` controls the ordinary input-field error budget, and
`--qoi-rel-eb` controls the derived speed error budget. Both constraints apply.
Remove `--limit 1000000` to process all particles. Use a new work directory for
each comparison, or add `--force` to overwrite an existing run.

`--xnyzip-tie-sort` is compatible with QoI processing. You can also add
`--xnyzip-linf-bound` to enforce the existing per-axis position bounds, as in
`run.sh`. The QoI choices here depend only on velocities; position compression
continues to use its existing bounds and options.

## Choose a quantity

| `--qoi` choice | Value calculated for each particle | Tolerance units |
| --- | --- | --- |
| `speed` | `sqrt(vx² + vy² + vz²)` | Velocity units |
| `kinetic-energy` | `0.5 * mass * (vx² + vy² + vz²)` | Mass × velocity units² |
| `vx-square` | `vx²` | Velocity units² |
| `vy-square` | `vy²` | Velocity units² |
| `vz-square` | `vz²` | Velocity units² |
| `none` | Disable QoI processing | — |

For kinetic energy, `--qoi-mass` is a single constant positive mass for all
particles. It defaults to `1.0`. Supply a mass consistent with your simulation's
velocity units; the frontend does not read masses from input metadata.

For example, preserve kinetic energy to an absolute tolerance of `10`:

```bash
python main.py roundtrip data/EXASKY-HACC-data-medium-size/ \
  --config config.yaml --work-dir particle_pipeline_runs/hacc_qoi_energy \
  --qoi kinetic-energy --qoi-mass 1.0 --qoi-abs-eb 10 \
  --limit 1000000 --metrics --clean-raw
```

To preserve a squared component instead, use, for example,
`--qoi vx-square --qoi-rel-eb 1e-4`.

## Choose the tolerance

Specify exactly one QoI tolerance:

| Option | Meaning |
| --- | --- |
| `--qoi-abs-eb T` | Every particle must satisfy `abs(Q_original - Q_reconstructed) <= T`. |
| `--qoi-rel-eb R` | Set `T = R * (max(Q_original) - min(Q_original))`, then enforce the same per-particle absolute bound. |

For example, if the original speeds range from `100` to `1100`, then
`--qoi-rel-eb 1e-4` resolves to a speed tolerance of `0.1` for every particle.
Range-relative tolerance is measured against the **original QoI range**, not
against each particle's value or a velocity component's range. With `--limit`,
the range is calculated using only the selected particles. For directory batch
input, each independent file has its own range and report.

Tolerances must be finite and non-negative. Zero requests exact preservation
of the evaluated QoI. A constant original QoI has range zero, so any relative
tolerance also resolves to zero; use an absolute tolerance if you want a
nonzero budget in that case. Tight or zero tolerances can substantially reduce
compression ratio.

The component error budgets from `--rel-eb`, `--vel-rel-eb`, or
`--vel-abs-eb` remain upper bounds. The frontend may tighten the SZO bounds to
meet the QoI tolerance. In the squared-component choices, only that component's
compression bound needs tightening.

## Configure it in YAML

Instead of adding QoI flags every time, edit the `qoi` section of the existing
`config.yaml`, keeping its `advanced` settings:

```yaml
qoi:
  quantity: speed
  abs_eb: null
  rel_eb: 1e-4
  mass: 1.0
```

The existing `advanced.pos_compressor: xnyzip` and
`advanced.vel_compressor: szo` must remain selected. The QoI section is optional
in other configuration files.

Command-line QoI options override YAML defaults. An explicit `--qoi-abs-eb`
replaces a YAML relative tolerance, and `--qoi-rel-eb` replaces a YAML absolute
tolerance. `--qoi none` disables QoI processing even when YAML enables it.
To disable it in YAML, set `quantity`, `abs_eb`, and `rel_eb` to `null`.

## Read the results

The console prints a block like:

```text
QoI: speed (per particle), abs_tolerance=0.233098
  SZO abs bounds: vx=0.134579, vy=0.134579, vz=0.134579
  max_abs_error=0.226251, corrected_particles=0/100000, correction_bytes=0, bound_ok=True
```

This example comes from the first 100,000 particles of the local HACC data,
with `--rel-eb 1e-3 --qoi speed --qoi-rel-eb 1e-4 --xnyzip-tie-sort`.

The work directory contains:

| Output | What it contains |
| --- | --- |
| `manifest.json` | Codec settings, resolved tolerance, selected SZO bounds, and complete encode-time QoI validation. |
| `compressed/` | Position and velocity streams, plus lossless QoI corrections when needed. |
| `reconstructed.h5` | Reconstructed particle data after applying corrections (`roundtrip` or `decompress`). |
| `metrics.json` | Detailed data metrics and a separate measurement of the final reconstructed QoI, when `--metrics` is used. |

Useful JSON entries are:

- `qoi.tolerance`: selected tolerance mode, original range, and resolved absolute tolerance.
- `qoi.requested_velocity_abs_bounds`: the original component error budgets.
- `qoi.compressor_velocity_abs_bounds`: the bounds actually supplied to SZO.
- `qoi.validation`: maximum error before and after corrections, violation
  counts, corrected particle count, and `satisfied`.
- `qoi.corrections`: sidecar path, compressed size, and corrected row count;
  `null` when no corrections are needed.
- `qoi.reconstruction_validation` in `metrics.json`: independently measured
  maximum error, RMSE, violation count, and `satisfied` on `reconstructed.h5`.

The reported maximum error is over all selected particles. QoI validation runs
even without `--metrics`; that flag adds an independent check of the final HDF5
file. Total package size and payload compression ratio include correction bytes
and manifest metadata. Timings include `qoi_prepare` and `qoi_validate`.

HACC has no IDs. XnYZip changes particle row order, and all velocity components
and QoI corrections follow that same order. The reconstructed HDF5 retains
particle associations. When comparing your own arrays, use the recorded
temporary position permutation before deleting preprocessing files, or request
`--metrics` during the roundtrip.

## Compress now, decompress later

To produce a package without keeping a reconstructed HDF5 file:

```bash
python main.py compress data/EXASKY-HACC-data-medium-size/ \
  --config config.yaml --work-dir particle_pipeline_runs/hacc_qoi_package \
  --qoi speed --qoi-rel-eb 1e-4 --limit 1000000
```

Compression internally decodes the velocity streams once to validate them and
package any required corrections. Later, use the ordinary decompression command:

```bash
python main.py decompress \
  --config config.yaml --work-dir particle_pipeline_runs/hacc_qoi_package \
  --clean-raw
```

Decompression reads the QoI settings from the manifest and applies corrections
automatically. It needs the manifest and compressed artifacts, including any
`qoi.corrections.zst` file. It does not need the original input, preprocessing
files, or QoI flags. Add `--force` when replacing an existing reconstruction.
Keep the package in its original work directory: its artifact paths follow the
repository's existing absolute-path convention.

`preprocess` also accepts the QoI options. Its report shows the selected bounds,
but the guarantee is established when `compress` or `roundtrip` validates the
actual codec output.

## Implementation and current scope

This module follows the estimate–validate–correct pattern in
[VLDB25-QPET.pdf](VLDB25-QPET.pdf), especially Sections 4, 5.3, and 5.4.
It implements a restricted frontend, rather than the paper's full symbolic
analyzer, point-wise bound storage, or compression-ratio auto-tuner.

It scans the selected velocities and derives conservative uniform bounds for
the named functions. For speed, a sufficient per-component bound is
`T / sqrt(3)`. For kinetic energy and squared components, it solves the
corresponding quadratic error inequality using the largest relevant original
magnitude. Each selected bound is capped by the ordinary component budget.

It then decodes the actual SZO streams and checks every particle against both
the QoI tolerance and the original velocity component budgets. Any failing row
is stored as its complete original velocity vector, losslessly compressed with
Zstandard. Corrections refer to reconstructed row indices and are applied after
HDF5 reconstruction. QoI scans and correction I/O use bounded blocks; the
underlying codecs retain their existing memory requirements. Encoding requires
an extra velocity decode and temporary space for those decoded arrays.

This version supports one named QoI per run, finite float32/float64 velocities,
and flat SZO velocity streams. Both HACC inputs and the existing HDF5/native
inputs are supported with XnYZip positions. Position tie sorting and XnYZip
position corrections are compatible. Lattice and structure-aware velocity
layouts, arbitrary expressions, regional aggregates, and per-particle masses
are outside the initial scope. Calculations and validation use float64;
speed uses `numpy.hypot`, and energy/squares use the formulas above. Inputs
whose derived QoIs overflow float64 are rejected.

## Verification

Run the focused tests with:

```bash
python -m unittest discover -s tests -p 'test_qoi.py' -v
```

The tests cover all five choices using real XnYZip/SZO codecs, relative and
absolute tolerance handling, zero-range data, canonical particle alignment,
mixed velocity dtypes, correction recovery without source data, and invalid
options/correction files.

The full regression suite currently has one pre-existing failure in
`test_batch_summary_reports_byte_weighted_field_group_crs`: ANSI color codes
interrupt the plain-text string it expects. This also fails on unmodified
`HEAD`; the other 135 tests pass with the QoI changes.

A local run on the first **1,000,000 HACC particles** with kinetic energy,
mass `1`, `--rel-eb 1e-3 --qoi-rel-eb 1e-4 --xnyzip-tie-sort
--xnyzip-linf-bound --metrics --clean-raw` passed both QoI and component checks.
Its resolved energy tolerance was **312.222**, maximum reconstructed energy
error **187.032**, and packaged compression ratio approximately **5.026**.
No QoI corrections were needed in that run. These are subset measurements;
they do not characterize the entire HACC dataset.
