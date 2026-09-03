# Particle Compressors

Particle Compressors is a command-line pipeline for lossy compression of
particle data stored in HDF5. One selected codec—SZO, SZ3, SPERR, QoZ, or
TTHRESH—compresses all six position and velocity fields. Particle IDs remain
lossless through pcodec. Roundtrip reconstruction preserves dataset paths,
dtypes, dataset attributes, and root HDF5 attributes.

## Input Format

The input must contain seven one-dimensional datasets with the same length.
Dataset basenames are matched case-insensitively.

| Logical field | Accepted dataset basenames | Required dtype |
| --- | --- | --- |
| ID | `id`, `particle_id`, `pid` | Any integer dtype |
| X | `x`, `posx`, `position_x` | Numeric |
| Y | `y`, `posy`, `position_y` | Numeric |
| Z | `z`, `posz`, `position_z` | Numeric |
| VX | `vx`, `velx`, `velocity_x` | `float32` or `float64` |
| VY | `vy`, `vely`, `velocity_y` | `float32` or `float64` |
| VZ | `vz`, `velz`, `velocity_z` | `float32` or `float64` |

Datasets may be inside HDF5 groups; matching uses the final path component.

## Requirements and Installation

- Python with development headers; Python 3.13 is known to work
- CMake and a C++ compiler for the local SPERR, QoZ, and TTHRESH tools
- Rust and Cargo for the local pcodec Python extension
- Initialized `tools/SZo`, `tools/SPERR`, `tools/QoZ`, `tools/tthresh`, and
  `tools/pcodec` submodules

```bash
git submodule update --init --recursive
conda create -n compressor python=3.13
conda activate compressor
bash install.sh
```

The installation script builds the local SPERR shared library and TTHRESH
executable, then installs the pinned Python dependencies, the local SZO,
SPERR, QoZ, and TTHRESH Python APIs, and the editable pcodec extension. SPERR
and TTHRESH are built without OpenMP because the pipeline already runs fields
in parallel. Their streams remain compatible with OpenMP-enabled builds.

## Quick Start

Run a complete SZO roundtrip with a relative error bound of `1e-3`:

```bash
python main.py roundtrip data/sample.h5 \
  --work-dir particle_pipeline_runs/sample-szo \
  --lossy-compressor szo \
  --rel-eb 1e-3 \
  --force
```

Select SZ3 for every lossy field with:

```bash
python main.py roundtrip data/sample.h5 \
  --work-dir particle_pipeline_runs/sample-sz3 \
  --lossy-compressor sz3 \
  --rel-eb 1e-3 \
  --force
```

Select SPERR's point-wise-error mode for every lossy field with:

```bash
python main.py roundtrip data/sample.h5 \
  --work-dir particle_pipeline_runs/sample-sperr \
  --lossy-compressor sperr \
  --rel-eb 1e-3 \
  --force
```

Select QoZ for every lossy field with:

```bash
python main.py roundtrip data/sample.h5 \
  --work-dir particle_pipeline_runs/sample-qoz \
  --lossy-compressor qoz \
  --rel-eb 1e-3 \
  --force
```

Select TTHRESH for every lossy field with:

```bash
python main.py roundtrip data/sample.h5 \
  --work-dir particle_pipeline_runs/sample-tthresh \
  --lossy-compressor tthresh \
  --rel-eb 1e-3 \
  --force
```

`--lossy-compressor` is the single codec selector and accepts `szo`, `sz3`,
`sperr`, `qoz`, or `tthresh`. The resulting compressed directory contains
`id.pco` plus one stream per lossy field. SZO streams use `.szo`, SZ3 streams
use `.psz`, SPERR streams use `.sperr`, QoZ streams use `.qoz`, and TTHRESH
streams use `.tthresh`.

Use a distinct work directory for each input and error-bound combination.
Existing outputs are rejected unless `--force` is supplied.

## Pipeline Commands

- `preprocess` exports raw fields and writes the initial manifest.
- `compress` preprocesses and writes the compressed package.
- `decompress` reconstructs an HDF5 file from an existing package.
- `roundtrip` compresses, reconstructs, and reports compression ratios and
  runtime. Add `--metrics` to also compute and write detailed quality metrics.

For a small environment check, add `--limit N`. Add `--clean-raw` to remove
the `preprocessed` and `decompressed` working directories after a roundtrip.

## Error Bounds

`--rel-eb` derives a separate absolute bound from each field range.
`--abs-eb` applies one default absolute bound. Position- and velocity-specific
options override the default:

- `--pos-rel-eb` or `--pos-abs-eb` for `x`, `y`, and `z`
- `--vel-rel-eb` or `--vel-abs-eb` for `vx`, `vy`, and `vz`

Relative and absolute options for the same field group are mutually
exclusive. Integer IDs are always reconstructed exactly; `--id-abs-eb` only
sets their expected metric bound.

Position data is converted to float32 compressor units before lossy coding.
`--position-scale` controls that conversion:

- `auto` uses the configured root attribute for integer positions when it is
  present, otherwise a scale of one.
- `raw` always uses a scale of one.
- `attr` requires the attribute named by `--position-scale-attr`.
- `value` requires an explicit `--position-scale-value`.

The manifest records preprocessing cast error and the adjusted compressor
bound for each position field.

TTHRESH receives relative bounds directly through its native relative-L2
error mode. This differs from the range-relative pointwise interpretation used
for the other lossy codecs, so TTHRESH reconstructions are not expected to
pass the final maximum-absolute-error check. With an absolute-bound CLI mode,
the TTHRESH adapter passes that value directly as an RMSE target. Flat fields
are padded to a balanced 3-D tensor; lattice fields retain their native 3-D
shape.

## Stable ID Sorting

Use `--sort` to stably sort particles by ascending ID before compression:

```bash
python main.py roundtrip data/sample.h5 \
  --work-dir particle_pipeline_runs/sample-sorted \
  --lossy-compressor szo \
  --sort \
  --force
```

The same permutation is applied to IDs, positions, and velocities, so every
reconstructed row retains particle correspondence. The sorted row order is
the package order. Metrics compare the matching source rows using the
temporary sort permutation, or by unique particle ID after `--clean-raw`.

## Periodic Lattice Layout

`--lattice-layout` enables an ID-derived dense 3-D transform before SZO, SZ3,
SPERR, QoZ, or TTHRESH compression and implies stable ID sorting:

```bash
python main.py roundtrip data/sample.h5 \
  --work-dir particle_pipeline_runs/sample-lattice \
  --lossy-compressor szo \
  --lattice-layout \
  --lattice-min-occupancy 0.8 \
  --force
```

The layout uses the root `nsidemesh` attribute and periodic particle IDs to
infer the dense lattice. If inference fails or occupancy is below
`--lattice-min-occupancy`, the pipeline retains the sorted flat-field path.
`--lattice-axis-search` tries all six dense-axis orders and stores the smallest
payload for each field; disable it with `--no-lattice-axis-search`.
The six floating-point fields are compressed and decompressed concurrently;
`--field-workers 0` selects up to six processes automatically, while a
positive value sets an explicit cap (use `--field-workers 1` for serial work).

Position fields may use a reversible periodic residual transform. Any wrap
offsets are stored losslessly with pcodec. The compressor bound is reduced by
the measured transform roundoff guard so the requested final bound remains
valid.

## Directory Batches

Pass a directory instead of a file to process its direct `.h5` children:

```bash
python main.py roundtrip data/snapshots \
  --work-dir particle_pipeline_runs/snapshots \
  --lossy-compressor sz3 \
  --file-workers 0 \
  --force
```

Directory discovery is non-recursive and matches the lowercase `.h5`
extension exactly. `--file-workers 0` selects up to 128 processes; a positive
value sets an explicit cap. Each input gets an isolated package directory.
The batch root receives `batch_metrics.json` with byte-weighted total and
field-group compression ratios, aggregate stage timings, throughput, per-file
statistics, and, when `--metrics` is supplied, per-field quality metrics.

To treat disjoint chunks as one particle set, add `--merge`:

```bash
python main.py roundtrip data/snapshots \
  --work-dir particle_pipeline_runs/snapshots-merged \
  --lossy-compressor szo \
  --sort \
  --lattice-layout \
  --merge \
  --force
```

The merge path checks all IDs exactly and rejects duplicate IDs within or
across chunks. It also requires matching field dtypes, dataset attributes, and
common root attributes. The concatenated source is written to
`WORK_DIR/merged/merged.h5` before preprocessing. Per-file root attributes are
normalized for the common file: `npart` and `npart_total` become the merged
particle count, while `rank` and `proc_size` become zero and one. Merge
provenance and elapsed time are recorded in the manifest. Without `--merge`,
directory processing retains the existing independent-file behavior.

## Package Contents

A compressed package contains:

- `manifest.json` with schema, attributes, bounds, codecs, ordering, layout,
  sizes, and timings
- `compressed/id.pco`
- six `.szo`, `.psz`, `.sperr`, `.qoz`, or `.tthresh` lossy field streams
- optional pcodec streams for lattice wrap offsets

Every completed roundtrip contains `reconstructed.h5`. `metrics.json` is
created only when `--metrics` is supplied. Detailed metrics
include maximum absolute error, MSE, RMSE, normalized RMSE, PSNR, bound
consistency, exact ID matching, component compression ratios, and total
payload compression ratio. Without `--metrics`, the console still reports
each component CR, total payload CR, and stage and end-to-end runtimes.

## Configuration

Advanced defaults live under `advanced` in `config.yaml`. Command-line options
override those defaults. Unknown configuration keys are rejected to catch
stale or misspelled settings.

## Code Structure

- `preprocess.py`, `compress.py`, and `decompress.py` orchestrate stages.
- `raw_codecs.py` adapts pcodec, QoZ, SPERR, SZ3, SZO, and TTHRESH streams.
- `lattice_layout.py` implements periodic dense transforms.
- `shaped_codecs.py` chooses adaptive 3-D codec layouts.
- `field_export.py`, `error_bounds.py`, and `hdf5_io.py` handle source
  conversion, bound selection, and HDF5 reconstruction.
- `manifest.py`, `metrics.py`, `batch.py`, and `runtime.py` contain package
  metadata, reporting, batch aggregation, and shared runtime utilities.

The scripts under `experiments/` explore payload and lattice-layout choices;
see `experiments/README.md` for entry points.
