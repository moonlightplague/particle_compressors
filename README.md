# Particle Compressors

Particle Compressors is a command-line pipeline for compressing particle data
stored in HDF5. It combines specialized codecs for each component of a particle
record, reconstructs the original HDF5 layout, and can calculate roundtrip error
and compression metrics. The pipeline preserves dataset names, dtypes, dataset attributes, and root HDF5
attributes in the reconstructed file.

## Input Format

The input must be an HDF5 file containing seven one-dimensional datasets with
the same length. Dataset basenames are matched case-insensitively using these
aliases:

| Logical field | Accepted dataset basenames | Required dtype |
| --- | --- | --- |
| ID | `id`, `particle_id`, `pid` | Any integer dtype |
| X | `x`, `posx`, `position_x` | Numeric |
| Y | `y`, `posy`, `position_y` | Numeric |
| Z | `z`, `posz`, `position_z` | Numeric |
| VX | `vx`, `velx`, `velocity_x` | `float32` or `float64` |
| VY | `vy`, `vely`, `velocity_y` | `float32` or `float64` |
| VZ | `vz`, `velz`, `velocity_z` | `float32` or `float64` |

Datasets may be located in HDF5 groups; matching uses only the final component
of each dataset path.

## Requirements

- Python with development headers; Python 3.13 is known to work
- A C++17 compiler, CMake, and Make for LCP
- Rust and Cargo for the local pcodec Python extension
- Git submodules initialized for `tools/LCP`, `tools/SZo`, and `tools/pcodec`

## Installation

Clone or initialize the native-code submodules:

```bash
git submodule update --init --recursive
```

From the repository root, the included installation script builds LCP and
installs the Python dependencies into the active Python environment:

```bash
bash install.sh
```


## Quick Start

Start with a limited roundtrip to validate the environment and input schema:


Run the full file with relative error bound of `1e-3`:

```bash
python main.py roundtrip data/sample.h5 \
  --config config.yaml \
  --work-dir particle_pipeline_runs \
  --rel-eb 1e-3 \
  --force 
```

Use a distinct work directory for each input/error-bound combination. Existing
outputs are rejected unless `--force` is supplied.

With `--pos-compressor lcp --vel-compressor sz3`, the compressed directory
contains `positions.lcp`, `id.pco`, `vx.psz`, `vy.psz`, and `vz.psz`. With
`--pos-compressor sz3 --vel-compressor lcp`, it contains `x.psz`, `y.psz`,
`z.psz`, `id.pco`, and `velocities.lcp`. Neither asymmetric configuration
stores an LCP order sidecar. If both triplets use LCP, `velocity_order.pco` is
also stored. Except for optional velocity LCP chunking, the pipeline does not
split fields into parts. Preprocessing, reconstruction, and metrics therefore
still load a complete field into memory at once. Manifests produced by the
older part-based format are not accepted by this format.

LCP sorts its input triplet before encoding it. In either asymmetric pipeline,
the pipeline adopts the LCP-sorted order as the reconstructed particle order
and applies the same temporary permutation to the ID and SZ3-compressed
triplet. Consequently no order sidecar is stored, while every reconstructed row
still contains the corresponding ID, position, and velocity. When both
triplets use LCP, position order is canonical and the independently sorted
velocity stream still requires `velocity_order.pco`.

## Chunked Velocity LCP

When both position and velocity triplets use LCP, independently compress
contiguous chunks of the position-ordered velocity rows with:

```bash
python main.py roundtrip data/sample.h5 \
  --work-dir particle_pipeline_runs_lcp_chunked \
  --pos-compressor lcp \
  --vel-compressor lcp \
  --lossless pcodec \
  --vel-chunk-size 4096 \
  --force
```

`--vel-chunk-size 0` disables chunking and retains the native monolithic LCP
stream. A positive value is only valid for the all-LCP pipeline. Each velocity
order index is then local to its chunk, so its unsigned range needs at most
`ceil(log2(chunk_size))` bits instead of `ceil(log2(particle_count))`. The raw
sidecar remains `int32`; pcodec bit-packs its non-negative range. The manifest
records both the theoretical width (`order_bits_per_particle`) and the actual
pcodec size (`compressed_bits_per_particle`).

`velocities.lcp` is a framed chunk container in this mode. Equal-sized chunks
are batched into native LCP calls with temporal prediction disabled, while a
short final chunk is encoded separately. Decompression validates every local
permutation, expands it to the corresponding position-ordered row range, and
then recombines the particle fields.

## Integer Compression

The default `--lossless pcodec` path is unchanged. Select SZO for IDs and the
optional permutation sidecar used by the all-LCP configuration with:

```bash
python main.py roundtrip data/sample.h5 \
  --work-dir particle_pipeline_runs_szo \
  --lossless szo \
  --szo-abs-eb 0.5 \
  --force
```

SZO is error-bounded rather than inherently lossless. The pipeline only accepts
an SZO absolute bound in the interval `[0, 1)`, where integer reconstruction
must be exact. SZO packages use `id.szo` and, when both triplets use LCP,
`velocity_order.szo`. Each stream stores a SHA-256 of the original
integer bytes; decompression fails before recombination if the reconstructed
bytes differ. Narrow and unsigned integer IDs are reversibly mapped to SZO's
supported `int32`/`int64` types. Integer mode uses SZO's Lorenzo/regression
algorithm; the default interpolation encoder is not used because its RLE/FSE
path is unstable on large, high-entropy permutation streams.


## Error Bounds

The default `--rel-eb 1e-3` applies to positions and velocities.
Position- and velocity-specific bounds can be supplied through
`--pos-abs-eb`, `--pos-rel-eb`, `--vel-abs-eb`, and `--vel-rel-eb`. A
field-class-specific value takes precedence over the global `--abs-eb` or
`--rel-eb`. LCP accepts one bound for the velocity triplet, so the strictest
derived `vx`/`vy`/`vz` bound is used. Do not set both absolute and relative
bounds for the same field class. IDs are always reconstructed exactly with
either pcodec or the checked SZO integer mode. `id_abs_eb` only defines the
expected ID error used by metrics and defaults to zero.
